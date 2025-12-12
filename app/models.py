from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# ---------------------------------------------------------
# UTILISATEURS
# ---------------------------------------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False) 
    role = db.Column(db.String(20), nullable=False, default='candidate') 
    
    created_exams = db.relationship('Exam', backref='creator', lazy=True)
    sessions = db.relationship('ExamSession', backref='candidate', lazy=True)

# ---------------------------------------------------------
# EXAMENS
# ---------------------------------------------------------
class Exam(db.Model):
    __tablename__ = 'exams'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(20), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    sessions = db.relationship('ExamSession', backref='exam', lazy=True)

# ---------------------------------------------------------
# SESSIONS
# ---------------------------------------------------------
class ExamSession(db.Model):
    __tablename__ = 'exam_sessions'
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    
    # Statistiques Globales
    final_score = db.Column(db.Float, default=0.0) 
    cheat_probability = db.Column(db.Float, default=0.0) 
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    
    # Relations
    predictions = db.relationship('PredictionLog', backref='session', lazy=True)
    alerts = db.relationship('CheatAlertSegment', backref='session', lazy=True)

# ---------------------------------------------------------
# LOGS DES PRÉDICTIONS
# ---------------------------------------------------------
class PredictionLog(db.Model):
    """
    Enregistre chaque fois que le modèle prédit '1' (Triche).
    """
    __tablename__ = 'prediction_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('exam_sessions.id'), nullable=False)
    
    # 0 = Normal, 1 = Triche
    prediction = db.Column(db.Integer, nullable=False) 
    
    # timestamp de la prédiction
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------------------------------------------------
# PÉRIODES D'ALERTE
# ---------------------------------------------------------
class CheatAlertSegment(db.Model):
    """
    Si le modèle prédit '1' trop souvent (ex: >60% du temps sur 10s),
    on crée une ligne ici.
    """
    __tablename__ = 'cheat_alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('exam_sessions.id'), nullable=False)
    
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    
    # Le taux de '1' détecté durant cette période
    avg_cheat_rate = db.Column(db.Float, default=0.0)