from unittest.mock import MagicMock, patch
from bantz.auth.token_store import TokenStore, TokenNotFoundError

def test_get_or_none_returns_none_on_exception():
    store = TokenStore()

    # Mock get() to raise an exception
    with patch.object(store, 'get', side_effect=Exception("Something went wrong")):
        result = store.get_or_none("gmail")
        assert result is None

    # Mock get() to raise TokenNotFoundError
    with patch.object(store, 'get', side_effect=TokenNotFoundError("Token not found")):
        result = store.get_or_none("gmail")
        assert result is None

def test_get_or_none_returns_credentials_on_success():
    store = TokenStore()
    mock_creds = MagicMock()

    # Mock get() to return valid credentials
    with patch.object(store, 'get', return_value=mock_creds):
        result = store.get_or_none("gmail")
        assert result == mock_creds
