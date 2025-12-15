from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

from config import Config
from models import db, User, Exam, ExamSession, CheatAlertSegment
from camera import VideoCamera

active_processors = {}

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = user.role
            session['username'] = user.username
            
            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('student_home'))
        else:
            flash('Identifiants incorrects', 'danger')
            
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role'] 
        
        hashed_pw = generate_password_hash(password)
        
        new_user = User(username=username, password=hashed_pw, role=role)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Compte créé ! Connectez-vous.', 'success')
            return redirect(url_for('login'))
        except:
            flash("Nom d'utilisateur déjà pris.", 'warning')
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    current_user_id = session['user_id']
    exams = Exam.query.filter_by(created_by=current_user_id).all()
    sessions = ExamSession.query.join(Exam).filter(Exam.created_by == current_user_id).all()
    
    return render_template('dashboard.html', exams=exams, sessions=sessions)

@app.route('/create_exam', methods=['GET', 'POST'])
def create_exam():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        title = request.form['title']
        code = request.form['code']

        existing_exam = Exam.query.filter_by(code=code).first()
        
        if existing_exam:
            flash("Ce code d'examen est déjà utilisé ! Veuillez en choisir un autre.", "danger")
            return redirect(url_for('create_exam'))
            
        
        new_exam = Exam(title=title, code=code, created_by=session['user_id'])
        
        try:
            db.session.add(new_exam)
            db.session.commit()
            flash('Examen créé avec succès !', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash(f"Erreur inattendue : {str(e)}", "danger")
            return redirect(url_for('create_exam'))
        
    return render_template('create_exam.html')

@app.route('/session_details/<int:session_id>')
def session_details(session_id):
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
        
    exam_session = ExamSession.query.get_or_404(session_id)
    alerts = CheatAlertSegment.query.filter_by(session_id=session_id).all()
    
    return render_template('session_details.html', exam_session=exam_session, alerts=alerts)

@app.route('/student')
def student_home():
    if session.get('role') != 'candidate':
        return redirect(url_for('login'))
    return render_template('join_exam.html')

@app.route('/join_exam', methods=['POST'])
def join_exam():
    code = request.form['code']
    exam = Exam.query.filter_by(code=code).first()
    
    if exam:
        new_session = ExamSession(
            user_id=session['user_id'],
            exam_id=exam.id,
            start_time=datetime.utcnow()
        )
        db.session.add(new_session)
        db.session.commit()
        
        return redirect(url_for('take_exam', session_id=new_session.id))
    else:
        flash("Code d'examen invalide.", 'danger')
        return redirect(url_for('student_home'))

@app.route('/exam/<int:session_id>')
def take_exam(session_id):
    if session.get('role') != 'candidate':
        return redirect(url_for('login'))
    
    # On prépare le processeur vidéo en mémoire
    if session_id not in active_processors:
        active_processors[session_id] = VideoCamera(app, session_id)
    
    current_session = ExamSession.query.get_or_404(session_id)
    return render_template('exam.html', exam_session=current_session, exam=current_session.exam)

# --- CORRECTION DE LA ROUTE API ---
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    session_id = data.get('session_id')
    image_base64 = data.get('image') 
    
    if not session_id or not image_base64:
        return jsonify({'status': 'error', 'msg': 'Missing data'}), 400

    if session_id in active_processors:
        processor = active_processors[session_id]
        result = processor.process_frame(image_base64) 
        return jsonify(result)
    else:
        try:
            active_processors[session_id] = VideoCamera(app, session_id)
            result = active_processors[session_id].process_frame(image_base64)
            return jsonify(result)
        except Exception as e:
            return jsonify({'status': 'error', 'msg': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)