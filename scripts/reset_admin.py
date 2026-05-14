import sys
sys.path.insert(0, ".")

from gateway.storage.db import SessionLocal
from gateway.storage.user_repo import hash_password
from gateway.storage.models import User

db = SessionLocal()

try:
    user = db.query(User).filter_by(email="admin@ppu.edu.ps").first()
    
    if user:
        new_hash = hash_password("admin123")
        user.hashed_password = new_hash
        db.commit()
        print("✅ ADMIN PASSWORD RESET SUCCESSFULLY!")
        print("Email    : admin@ppu.edu.ps")
        print("Password : admin123")
        print("You can now login.")
    else:
        print("❌ Could not find admin@ppu.edu.ps")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()