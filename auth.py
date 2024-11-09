from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import config

Base = declarative_base()
bcrypt = Bcrypt()
login_manager = LoginManager()

class User(Base, UserMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

engine = create_engine(config.DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

@login_manager.user_loader
def load_user(user_id):
    session = Session()
    return session.query(User).get(int(user_id))

def init_auth(app):
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    bcrypt.init_app(app)

def create_user(username, password):
    session = Session()
    user = User(username=username)
    user.set_password(password)
    session.add(user)
    session.commit()
    session.close()

def authenticate_user(username, password):
    session = Session()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if user and user.check_password(password):
        return user
    return None