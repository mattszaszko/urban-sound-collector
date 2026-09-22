"""Tests for admin / viewer session auth."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from web import auth


class AuthRoleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {
            "SECRET_KEY": os.environ.get("SECRET_KEY"),
            "USC_PASSWORD": os.environ.get("USC_PASSWORD"),
            "USC_VIEWER_PASSWORD": os.environ.get("USC_VIEWER_PASSWORD"),
        }
        os.environ["SECRET_KEY"] = "unit-test-secret-key"
        os.environ["USC_PASSWORD"] = "admin-secret"
        os.environ["USC_VIEWER_PASSWORD"] = "viewer-secret"

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _request_with_cookie(self, token: str) -> MagicMock:
        req = MagicMock()
        req.cookies = {auth.SESSION_COOKIE: token}
        return req

    def test_verify_login_admin_and_viewer(self) -> None:
        self.assertEqual(auth.verify_login("admin", "admin-secret"), "admin")
        self.assertEqual(auth.verify_login("viewer", "viewer-secret"), "viewer")
        self.assertIsNone(auth.verify_login("admin", "viewer-secret"))
        self.assertIsNone(auth.verify_login("viewer", "admin-secret"))
        self.assertIsNone(auth.verify_login("admin", "wrong"))
        self.assertIsNone(auth.verify_login("nope", "admin-secret"))

    def test_viewer_login_disabled_when_password_empty(self) -> None:
        os.environ.pop("USC_VIEWER_PASSWORD", None)
        self.assertIsNone(auth.verify_login("viewer", "anything"))

    def test_session_role_and_password_rotation(self) -> None:
        token = auth.make_session_cookie("admin-secret", "admin")
        req = self._request_with_cookie(token)
        self.assertTrue(auth.is_authenticated(req))
        self.assertEqual(auth.get_role(req), "admin")
        self.assertTrue(auth.is_admin(req))
        self.assertFalse(auth.is_viewer(req))

        viewer_token = auth.make_session_cookie("viewer-secret", "viewer")
        viewer_req = self._request_with_cookie(viewer_token)
        self.assertEqual(auth.get_role(viewer_req), "viewer")
        self.assertTrue(auth.is_viewer(viewer_req))
        self.assertFalse(auth.is_admin(viewer_req))

        os.environ["USC_PASSWORD"] = "rotated-admin"
        self.assertFalse(auth.is_authenticated(req))
        self.assertIsNone(auth.get_role(req))

    def test_legacy_cookie_rejected(self) -> None:
        legacy = auth._serializer().dumps({"authed": True, "pw_hash": "x"})
        req = self._request_with_cookie(legacy)
        self.assertFalse(auth.is_authenticated(req))
        self.assertIsNone(auth.get_role(req))


if __name__ == "__main__":
    unittest.main()
