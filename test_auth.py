import unittest
from database import Database
from sqlalchemy.orm import Session
from passlib.context import CryptContext 

# Import the functions you want to test
from auth import (
    verify_password,
    get_password_hash,
    authenticate_user,
    create_user
)

class TestAuth(unittest.TestCase):
    
    def setUp(self):
        # We need a live database for auth functions to work
        self.db = Database()
        # Create tables fresh for *every* test to ensure isolation
        self.db.Base.metadata.create_all(bind=self.db.engine)
        self.session = next(self.db.get_db())

    def tearDown(self):
        self.session.close()
        # Drop all tables after *every* test to ensure isolation
        self.db.Base.metadata.drop_all(bind=self.db.engine)

    def test_create_user_and_hashing(self):
        # Test creating a user
        plain_password = "mypassword123"
        user = create_user(
            self.session, 
            name="Test Auth User", 
            email="auth@example.com", 
            password=plain_password
        )
        
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "auth@example.com")
        
        # Test that the password was actually hashed
        self.assertNotEqual(user.hashed_password, plain_password)
        self.assertTrue(verify_password(plain_password, user.hashed_password))

    def test_authenticate_user_success(self):
        # 1. Create a user first
        plain_password = "correct_password"
        create_user(
            self.session, 
            name="Login User", 
            email="login@example.com", 
            password=plain_password
        )
        
        # 2. Try to authenticate
        authenticated_user = authenticate_user(
            self.session, 
            email="login@example.com", 
            password=plain_password
        )
        
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.name, "Login User")

    def test_authenticate_user_fail_wrong_password(self):
        # 1. Create a user
        plain_password = "correct_password"
        create_user(
            self.session, 
            name="Login User", 
            email="login@example.com", 
            password=plain_password
        )
        
        # 2. Try to authenticate with wrong password
        authenticated_user = authenticate_user(
            self.session, 
            email="login@example.com", 
            password="WRONG_PASSWORD"
        )
        
        self.assertFalse(authenticated_user) # Should return False

    def test_authenticate_user_fail_no_user(self):
        # Try to authenticate a user that doesn't exist
        authenticated_user = authenticate_user(
            self.session, 
            email="nobody@example.com", 
            password="password"
        )
        
        self.assertFalse(authenticated_user) # Should return False


if __name__ == '__main__':
    unittest.main()