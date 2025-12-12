from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Import de la configuration et des modèles
from config import Config
from models import db, User, Exam, ExamSession, CheatAlertSegment
from camera import VideoCamera

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

with app.app_context():
    db.create_all()

# --- FONCTION GÉNÉRATEUR POUR LE STREAMING ---
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            # Stockage des infos dans la session navigateur
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
        
        # Hachage du mot de passe pour la sécurité
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
    
    exams = Exam.query.all()
    sessions = ExamSession.query.all()
    
    return render_template('dashboard.html', exams=exams, sessions=sessions)

@app.route('/create_exam', methods=['GET', 'POST'])
def create_exam():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        title = request.form['title']
        code = request.form['code']
        
        new_exam = Exam(title=title, code=code, created_by=session['user_id'])
        db.session.add(new_exam)
        db.session.commit()
        flash('Examen créé avec succès !', 'success')
        return redirect(url_for('dashboard'))
        
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
        
    current_session = ExamSession.query.get_or_404(session_id)
    return render_template('exam.html', exam_session=current_session, exam=current_session.exam)

@app.route('/video_feed/<int:session_id>')
def video_feed(session_id):
    return Response(gen(VideoCamera(app, session_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)