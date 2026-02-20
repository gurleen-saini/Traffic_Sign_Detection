from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    password = password[:72]   # ⭐ bcrypt max limit fix
    return pwd_context.hash(password)

def verify_password(password, hashed):
    password = password[:72]   # ⭐ same limit while checking
    return pwd_context.verify(password, hashed)
