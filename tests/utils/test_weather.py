"""Tests for weather utilities."""

from datetime import date
from unittest.mock import Mock

import pandas as pd

from utils.weather import (
    WeatherDataCollector,
    build_multiscale_windows,
    fetch_weather_window,
    make_weather_id,
)


class TestFetchWeatherWindow:
    def test_returns_dataframe_with_weather_data(self, mocker):
        """fetch_weather_window returns DataFrame with expected columns."""
        # Mock Open-Meteo response
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2023-05-01", "2023-05-02", "2023-05-03"],
                "temperature_2m_mean": [10.5, 11.2, 9.8],
                "temperature_2m_max": [15.0, 16.0, 14.5],
                "temperature_2m_min": [6.0, 7.0, 5.5],
                "wind_speed_10m_max": [25.0, 30.0, 20.0],
            }
        }
        mock_response.raise_for_status = Mock()
        mocker.patch("requests.get", return_value=mock_response)

        result = fetch_weather_window(
            latitude=27.99,
            longitude=86.93,
            target_date=date(2023, 5, 3),
            window_days=3,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "temperature_2m_mean" in result.columns
        assert "date" in result.columns

    def test_handles_api_error_gracefully(self, mocker):
        """Returns empty DataFrame on API error."""
        mocker.patch("requests.get", side_effect=Exception("API Error"))

        result = fetch_weather_window(
            latitude=27.99,
            longitude=86.93,
            target_date=date(2023, 5, 3),
            window_days=3,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestBuildMultiscaleWindows:
    def test_returns_three_scales(self):
        """Returns dict with 7d, 30d, 90d keys."""
        # Create raw weather data (90 days)
        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame(
            {
                "date": dates,
                "temperature_2m_mean": range(90),
                "wind_speed_10m_max": range(90),
            }
        )

        result = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))

        assert "7d" in result
        assert "30d" in result
        assert "90d" in result

    def test_7d_window_has_7_rows(self):
        """7-day window at full resolution."""
        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame(
            {
                "date": dates,
                "temperature_2m_mean": range(90),
            }
        )

        result = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))

        assert len(result["7d"]) == 7

    def test_30d_window_has_10_aggregates(self):
        """30-day window aggregated to 10 three-day buckets."""
        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame(
            {
                "date": dates,
                "temperature_2m_mean": range(90),
            }
        )

        result = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))

        assert len(result["30d"]) == 10

    def test_90d_window_has_9_aggregates(self):
        """90-day window aggregated to 9 ten-day buckets."""
        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame(
            {
                "date": dates,
                "temperature_2m_mean": range(90),
            }
        )

        result = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))

        assert len(result["90d"]) == 9


class TestWeatherCaching:
    def test_weather_id_format(self):
        """Weather ID follows {peakid}_{date} format."""
        weather_id = make_weather_id("EVER", date(2023, 5, 15))
        assert weather_id == "EVER_2023-05-15"

    def test_weather_id_handles_string_date(self):
        """Weather ID works with string dates too."""
        weather_id = make_weather_id("EVER", "2023-05-15")
        assert weather_id == "EVER_2023-05-15"

    def test_collector_add_and_has_weather(self, tmp_path):
        """Can add weather data and check if it exists."""
        collector = WeatherDataCollector(output_dir=tmp_path)

        # Create multi-scale windows
        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame({
            "date": dates,
            "temperature_2m_mean": range(90),
        })
        windows = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))

        # Initially should not have weather
        assert not collector.has_weather("EVER", date(2023, 5, 15))

        # Add weather
        weather_id = collector.add(windows, "EVER", date(2023, 5, 15))

        # Now should have weather
        assert collector.has_weather("EVER", date(2023, 5, 15))
        assert weather_id == "EVER_2023-05-15"

    def test_collector_save_and_load(self, tmp_path):
        """Can save weather data and load it back."""
        collector = WeatherDataCollector(output_dir=tmp_path)

        # Create and add weather data
        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame({
            "date": dates,
            "temperature_2m_mean": range(90),
        })
        windows = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))
        collector.add(windows, "EVER", date(2023, 5, 15))

        # Save to CSV
        output_path = collector.save()
        assert output_path.exists()

        # Load into new collector
        new_collector = WeatherDataCollector(output_dir=tmp_path)
        new_collector.load_existing()

        assert new_collector.has_weather("EVER", date(2023, 5, 15))
        assert len(new_collector) == 1

    def test_collector_deduplicates(self, tmp_path):
        """Adding same peak+date twice should not create duplicates."""
        collector = WeatherDataCollector(output_dir=tmp_path)

        dates = pd.date_range(end="2023-05-15", periods=90, freq="D")
        raw_df = pd.DataFrame({
            "date": dates,
            "temperature_2m_mean": range(90),
        })
        windows = build_multiscale_windows(raw_df, target_date=date(2023, 5, 15))

        collector.add(windows, "EVER", date(2023, 5, 15))
        collector.add(windows, "EVER", date(2023, 5, 15))

        assert len(collector) == 1
