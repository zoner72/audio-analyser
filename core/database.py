from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class AudioSegment(Base):
    __tablename__ = 'audio_segments'
    
    id = Column(Integer, primary_key=True)
    original_file = Column(String)
    segment_file = Column(String)
    start_time = Column(Float)
    end_time = Column(Float)
    classification = Column(String)
    datetime = Column(DateTime)
    features = Column(String)  # JSON string of features

class Database:
    def __init__(self):
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'audio_analysis.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def save_segment(self, segment_data, classification):
        try:
            segment = AudioSegment(
                original_file=segment_data['original_file'],
                segment_file=segment_data['segment_file'],
                start_time=segment_data['start_time'],
                end_time=segment_data['end_time'],
                classification=classification,
                datetime=segment_data['datetime'],
                features=str(segment_data['features'])
            )
            self.session.add(segment)
            self.session.commit()
            return True
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            self.session.rollback()
            return False

    def get_all_segments(self):
        return self.session.query(AudioSegment).all()
