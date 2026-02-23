"""Tests for geocoding utilities."""

import pandas as pd
import pytest

from utils.geocoding import geocode_peak, geocode_peaks_batch


class TestGeocodePeak:
    def test_returns_coordinates_for_known_peak(self, mocker):
        """Nominatim returns coords for Everest."""
        mock_response = mocker.Mock()
        mock_response.json.return_value = [
            {"lat": "27.9881", "lon": "86.9250", "display_name": "Mount Everest"}
        ]
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=mock_response)

        result = geocode_peak("Everest", "Khumbu Himal")

        assert result is not None
        lat, lon = result
        assert 27.0 < lat < 29.0
        assert 86.0 < lon < 88.0

    def test_returns_none_when_not_found(self, mocker):
        """Returns None when Nominatim finds nothing."""
        mock_response = mocker.Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=mock_response)

        result = geocode_peak("NonexistentPeak123", "Unknown")

        assert result is None


class TestGeocodePeaksBatch:
    def test_adds_lat_lon_columns(self, mocker):
        """Batch geocoding adds latitude/longitude columns."""
        mocker.patch(
            "utils.geocoding.geocode_peak",
            side_effect=[
                (27.99, 86.93),  # Everest
                (28.55, 83.82),  # Annapurna
                None,  # Unknown peak
            ],
        )

        df = pd.DataFrame(
            {
                "peakid": ["EVER", "ANN1", "UNKN"],
                "pkname": ["Everest", "Annapurna I", "Unknown Peak"],
                "himal": ["Khumbu Himal", "Annapurna Himal", "Unknown"],
            }
        )

        result = geocode_peaks_batch(df)

        assert "latitude" in result.columns
        assert "longitude" in result.columns
        assert result.loc[0, "latitude"] == 27.99
        assert result.loc[0, "longitude"] == 86.93
        assert pd.isna(result.loc[2, "latitude"])
