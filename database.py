from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Get database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()

class StockData(Base):
    """Model for storing historical stock data"""
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)

class UserPreference(Base):
    """Model for storing user preferences"""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    is_favorite = Column(Boolean, default=False)
    last_viewed = Column(DateTime, default=datetime.utcnow)

# Create database tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def cache_stock_data(symbol: str, historical_data):
    """Cache stock data in the database"""
    db = next(get_db())
    
    try:
        # Convert historical data to database records
        for date, row in historical_data.iterrows():
            stock_data = StockData(
                symbol=symbol,
                date=date,
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                volume=row['Volume']
            )
            db.add(stock_data)
        
        db.commit()
    except Exception as e:
        db.rollback()
        raise e

def get_cached_stock_data(symbol: str, days: int = 365):
    """Get cached stock data from database"""
    db = next(get_db())
    from datetime import timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    return db.query(StockData)\
             .filter(StockData.symbol == symbol,
                    StockData.date >= cutoff_date)\
             .order_by(StockData.date).all()

def update_user_preference(symbol: str, is_favorite: bool = False):
    """Update user preference for a stock"""
    db = next(get_db())
    
    pref = db.query(UserPreference)\
             .filter(UserPreference.symbol == symbol)\
             .first()
    
    if not pref:
        pref = UserPreference(symbol=symbol)
        db.add(pref)
    
    pref.is_favorite = is_favorite
    pref.last_viewed = datetime.utcnow()
    db.commit()
