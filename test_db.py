import unittest
from database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        self.db.create_tables()
        self.session = next(self.db.get_db())

    def tearDown(self):
        # Optionally drop tables or clean up here to ensure test isolation
        self.session.close()

    def test_create_user(self):
        
        new_user = self.db.User(
            name="Test User",
            email="BqC2o@example.com",
            hashed_password="hashed_password",
            is_active=True
        )
        self.session.add(new_user)
        self.session.commit()
        user_in_db = self.session.query(self.db.User).filter_by(name="Test User").one_or_none()
        self.assertIsNotNone(user_in_db)

    def test_create_ai_service(self):
        new_service = self.db.AiService(
            name="Test Service",
            description="This is a test AI service.",
            is_active=True
        )
        self.session.add(new_service)
        self.session.commit()
        service_in_db = self.session.query(self.db.AiService).filter_by(name="Test Service").one_or_none()
        self.assertIsNotNone(service_in_db)


if __name__ == '__main__':
    unittest.main()
