"""Tests for weather utilities."""

from datetime import date
from unittest.mock import Mock

import pandas as pd

from utils.weather import (
    build_multiscale_windows,
    fetch_weather_window,
    get_cached_weather,
    get_weather_cache_path,
    save_weather_cache,
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
    def test_cache_path_format(self):
        """Cache path follows {peakid}_{date}.csv format."""
        path = get_weather_cache_path("EVER", date(2023, 5, 15))
        assert path.name == "EVER_2023-05-15.csv"

    def test_save_and_load_cache(self, tmp_path):
        """Can save and load weather cache."""
        cache_dir = tmp_path / "weather"
        df = pd.DataFrame(
            {
                "date": ["2023-05-15"],
                "temperature_2m_mean": [10.5],
            }
        )

        save_weather_cache(df, "EVER", date(2023, 5, 15), cache_dir=cache_dir)
        loaded = get_cached_weather("EVER", date(2023, 5, 15), cache_dir=cache_dir)

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded["temperature_2m_mean"].iloc[0] == 10.5

    def test_returns_none_when_no_cache(self, tmp_path):
        """Returns None when cache doesn't exist."""
        cache_dir = tmp_path / "weather"
        result = get_cached_weather("NOTEXIST", date(2023, 5, 15), cache_dir=cache_dir)
        assert result is None
