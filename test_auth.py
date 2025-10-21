import unittest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from auth import (
    verify_password,
    get_password_hash,
    authenticate_user,
    create_user,
    get_user_by_email,
    create_access_token,
    get_current_user,
    deactivate_user,
    activate_user
)
from config import settings


class TestAuth(unittest.TestCase):
    
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
            self.session.query(Database.User).delete()
            self.session.commit()
        except:
            self.session.rollback()
        finally:
            self.session.close()

    def test_password_hashing_and_verification(self):
        """Test password hashing and verification functions"""
        plain_password = "secure_password_123"
        
        # Test hashing
        hashed_password = get_password_hash(plain_password)
        self.assertIsNotNone(hashed_password)
        self.assertNotEqual(hashed_password, plain_password)
        self.assertTrue(hashed_password.startswith("$2b$"))  # bcrypt format
        
        # Test verification
        self.assertTrue(verify_password(plain_password, hashed_password))
        self.assertFalse(verify_password("wrong_password", hashed_password))
        self.assertFalse(verify_password("", hashed_password))

    def test_create_user_success(self):
        """Test successful user creation"""
        user_data = {
            "name": "Test Auth User",
            "email": "auth.success@example.com",
            "password": "mypassword123"
        }
        
        user = create_user(self.session, **user_data)
        
        self.assertIsNotNone(user)
        self.assertEqual(user.name, user_data["name"])
        self.assertEqual(user.email, user_data["email"])
        self.assertNotEqual(user.hashed_password, user_data["password"])
        self.assertTrue(user.is_active)
        self.assertIsNotNone(user.created_at)
        
        # Verify password works
        self.assertTrue(verify_password(user_data["password"], user.hashed_password))

    def test_create_user_duplicate_email(self):
        """Test that duplicate email registration fails"""
        user_data = {
            "name": "First User",
            "email": "duplicate@example.com",
            "password": "password123"
        }
        
        # Create first user
        user1 = create_user(self.session, **user_data)
        self.assertIsNotNone(user1)
        
        # Try to create second user with same email
        with self.assertRaises(HTTPException) as context:
            create_user(
                self.session, 
                name="Second User", 
                email="duplicate@example.com", 
                password="different_password"
            )
        
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("already registered", context.exception.detail.lower())

    def test_authenticate_user_success(self):
        """Test successful user authentication"""
        # Create a user first
        user_data = {
            "name": "Login User",
            "email": "login.success@example.com", 
            "password": "correct_password"
        }
        create_user(self.session, **user_data)
        
        # Test authentication
        authenticated_user = authenticate_user(
            self.session, 
            email=user_data["email"], 
            password=user_data["password"]
        )
        
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.name, user_data["name"])
        self.assertEqual(authenticated_user.email, user_data["email"])
        self.assertTrue(authenticated_user.is_active)

    def test_authenticate_user_wrong_password(self):
        """Test authentication failure with wrong password"""
        user_data = {
            "name": "Login User",
            "email": "login.fail@example.com",
            "password": "correct_password"
        }
        create_user(self.session, **user_data)
        
        # Test with wrong password
        authenticated_user = authenticate_user(
            self.session, 
            email=user_data["email"], 
            password="WRONG_PASSWORD"
        )
        
        self.assertFalse(authenticated_user)

    def test_authenticate_user_nonexistent_email(self):
        """Test authentication failure with non-existent email"""
        authenticated_user = authenticate_user(
            self.session, 
            email="nonexistent@example.com", 
            password="any_password"
        )
        
        self.assertFalse(authenticated_user)

    def test_authenticate_user_inactive_user(self):
        """Test authentication failure with inactive user"""
        # Create active user
        user_data = {
            "name": "Inactive User",
            "email": "inactive@example.com",
            "password": "password123"
        }
        user = create_user(self.session, **user_data)
        
        # Deactivate user
        deactivated_user = deactivate_user(self.session, user)
        self.assertFalse(deactivated_user.is_active)
        
        # Try to authenticate inactive user
        authenticated_user = authenticate_user(
            self.session,
            email=user_data["email"],
            password=user_data["password"]
        )
        
        self.assertFalse(authenticated_user)

    def test_get_user_by_email(self):
        """Test retrieving user by email"""
        user_data = {
            "name": "Get User Test",
            "email": "getuser@example.com",
            "password": "password123"
        }
        created_user = create_user(self.session, **user_data)
        
        # Retrieve user
        retrieved_user = get_user_by_email(self.session, user_data["email"])
        
        self.assertIsNotNone(retrieved_user)
        self.assertEqual(retrieved_user.id, created_user.id)
        self.assertEqual(retrieved_user.email, user_data["email"])
        
        # Test non-existent user
        non_existent_user = get_user_by_email(self.session, "nonexistent@example.com")
        self.assertIsNone(non_existent_user)

    def test_user_activation_deactivation(self):
        """Test user activation and deactivation"""
        user_data = {
            "name": "Activation Test User",
            "email": "activation@example.com",
            "password": "password123"
        }
        user = create_user(self.session, **user_data)
        self.assertTrue(user.is_active)
        
        # Deactivate user
        deactivated_user = deactivate_user(self.session, user)
        self.assertFalse(deactivated_user.is_active)
        
        # Reactivate user
        activated_user = activate_user(self.session, deactivated_user)
        self.assertTrue(activated_user.is_active)

    def test_create_access_token(self):
        """Test JWT token creation"""
        test_data = {"sub": "test@example.com", "user_id": 123}
        
        # Test token creation with default expiration
        token = create_access_token(data=test_data)
        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)
        
        # Test token creation with custom expiration
        from datetime import timedelta
        custom_token = create_access_token(
            data=test_data, 
            expires_delta=timedelta(hours=2)
        )
        self.assertIsNotNone(custom_token)

    def test_password_complexity_handling(self):
        """Test handling of various password scenarios"""
        test_cases = [
            "short",           # Very short password
            "very_long_password_that_exceeds_normal_limits_but_should_still_work",  # Long password
            "password with spaces",  # Password with spaces
            "Special@Chars#123!",  # Complex password
        ]
        
        for password in test_cases:
            with self.subTest(password=password):
                hashed = get_password_hash(password)
                self.assertIsNotNone(hashed)
                self.assertTrue(verify_password(password, hashed))
                self.assertFalse(verify_password(password + "wrong", hashed))


class TestAuthEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.session = self.TestSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, 
            bind=create_engine('sqlite:///:memory:')
        )()
        Database.Base.metadata.create_all(bind=self.TestSessionLocal.bind)
    
    def tearDown(self):
        self.session.rollback()
        self.session.close()
    
    def test_empty_password(self):
        """Test handling of empty password"""
        with self.assertRaises(Exception):
            create_user(
                self.session,
                name="Empty Password User",
                email="empty@example.com",
                password=""
            )
    
    def test_sql_injection_attempt(self):
        """Test handling of potential SQL injection in email"""
        malicious_email = "test@example.com' OR '1'='1"
        
        # This should not crash and should handle safely
        user = get_user_by_email(self.session, malicious_email)
        self.assertIsNone(user)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)