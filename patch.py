import sys
from unittest.mock import patch, MagicMock

import bantz.core.gps_server as mod

def test():
    data = {"lat": 12.34, "lon": 56.78, "accuracy": 10.5, "timestamp": "2023-01-01T00:00:00Z"}

    with patch("sys.modules", dict(sys.modules)):
        mock_places = MagicMock()
        mock_places_mod = MagicMock()
        mock_places_mod.places = mock_places
        sys.modules["bantz.core.places"] = mock_places_mod

        srv = mod.GPSServer(port=9999)
        srv._save_location(data)

        print("Calls:", mock_places.update_gps.call_args_list)

test()
