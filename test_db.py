import unittest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from config import settings


class TestDatabase(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test database once for all tests"""
        # Use test database URL or create in-memory SQLite for tests
        cls.test_db_url = os.getenv('TEST_DATABASE_URL', 'sqlite:///:memory:')
        
        # Create test engine and session
        cls.engine = create_engine(cls.test_db_url)
        cls.TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        
        # Create all tables
        Database.Base.metadata.create_all(bind=cls.engine)

    def setUp(self):
        """Set up fresh session for each test"""
        self.session = self.TestSessionLocal()
        
    def tearDown(self):
        """Clean up after each test"""
        # Rollback any uncommitted changes
        self.session.rollback()
        
        # Clean up test data
        try:
            self.session.query(Database.AiService).delete()
            self.session.query(Database.User).delete()
            self.session.commit()
        except:
            self.session.rollback()
        finally:
            self.session.close()

    def test_create_user(self):
        """Test user creation with valid data"""
        new_user = Database.User(
            name="Test User",
            email="testuser@example.com",  # Fixed email format
            hashed_password="hashed_password_123",
            is_active=True
        )
        
        self.session.add(new_user)
        self.session.commit()
        
        # Verify user was created
        user_in_db = self.session.query(Database.User).filter_by(
            email="testuser@example.com"
        ).first()
        
        self.assertIsNotNone(user_in_db)
        self.assertEqual(user_in_db.name, "Test User")
        self.assertEqual(user_in_db.email, "testuser@example.com")
        self.assertTrue(user_in_db.is_active)
        self.assertIsNotNone(user_in_db.created_at)

    def test_create_user_duplicate_email(self):
        """Test that duplicate emails are prevented"""
        user1 = Database.User(
            name="User One",
            email="duplicate@example.com",
            hashed_password="hash1",
            is_active=True
        )
        
        user2 = Database.User(
            name="User Two", 
            email="duplicate@example.com",  # Same email
            hashed_password="hash2",
            is_active=True
        )
        
        self.session.add(user1)
        self.session.commit()
        
        # Second user with same email should raise integrity error
        self.session.add(user2)
        with self.assertRaises(Exception):  # Should raise IntegrityError
            self.session.commit()
        
        self.session.rollback()

    def test_create_ai_service(self):
        """Test AI service creation"""
        new_service = Database.AiService(
            name="Test AI Service",
            description="This is a test AI service for unit testing.",
            is_active=True
        )
        
        self.session.add(new_service)
        self.session.commit()
        
        # Verify service was created
        service_in_db = self.session.query(Database.AiService).filter_by(
            name="Test AI Service"
        ).first()
        
        self.assertIsNotNone(service_in_db)
        self.assertEqual(service_in_db.name, "Test AI Service")
        self.assertEqual(service_in_db.description, "This is a test AI service for unit testing.")
        self.assertTrue(service_in_db.is_active)
        self.assertIsNotNone(service_in_db.created_at)

    def test_user_inactive_by_default(self):
        """Test that new users are active by default"""
        new_user = Database.User(
            name="Inactive Test",
            email="inactive@example.com",
            hashed_password="hashed_pass"
            # is_active not set, should use default
        )
        
        self.session.add(new_user)
        self.session.commit()
        
        user_in_db = self.session.query(Database.User).filter_by(
            email="inactive@example.com"
        ).first()
        
        self.assertTrue(user_in_db.is_active)  # Should be True by default

    def test_ai_service_inactive_by_default(self):
        """Test that new AI services are active by default"""
        new_service = Database.AiService(
            name="Default Active Service",
            description="Testing default active status"
            # is_active not set, should use default
        )
        
        self.session.add(new_service)
        self.session.commit()
        
        service_in_db = self.session.query(Database.AiService).filter_by(
            name="Default Active Service"
        ).first()
        
        self.assertTrue(service_in_db.is_active)  # Should be True by default

    def test_database_connection(self):
        """Test that database connection works"""
        # This tests the actual connection to the database
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                self.assertEqual(result.scalar(), 1)
        except Exception as e:
            self.fail(f"Database connection test failed: {e}")

    def test_user_repr(self):
        """Test User __repr__ method"""
        user = Database.User(
            name="Repr Test",
            email="repr@example.com",
            hashed_password="hash",
            is_active=True
        )
        
        repr_string = repr(user)
        self.assertIn("Repr Test", repr_string)
        self.assertIn("repr@example.com", repr_string)
        self.assertIn("True", repr_string)

    def test_ai_service_repr(self):
        """Test AiService __repr__ method"""
        service = Database.AiService(
            name="Repr Service",
            description="Test repr",
            is_active=False
        )
        
        repr_string = repr(service)
        self.assertIn("Repr Service", repr_string)
        self.assertIn("False", repr_string)


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database operations"""
    
    def setUp(self):
        self.session = next(Database.get_db())
        
    def tearDown(self):
        self.session.rollback()
        self.session.close()
    
    def test_get_db_dependency(self):
        """Test that get_db dependency works correctly"""
        db_gen = Database.get_db()
        db_session = next(db_gen)
        
        self.assertIsNotNone(db_session)
        
        # Should be able to perform operations
        user_count = db_session.query(Database.User).count()
        self.assertIsInstance(user_count, int)
        
        # Clean up
        try:
            next(db_gen)  # This should raise StopIteration
        except StopIteration:
            pass  # Expected behavior


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)