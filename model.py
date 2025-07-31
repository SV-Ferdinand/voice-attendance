from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import json
import uuid
from sqlalchemy import Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
import enum

db = SQLAlchemy()

class AttendanceStatus(enum.Enum):
    HADIR = "HADIR"
    ALPHA = "ALPHA"
    IZIN = "IZIN"
    SAKIT = "SAKIT"

class VoiceQuality(enum.Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"

class AdminRole(enum.Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    TEACHER = "teacher"

class Admin(db.Model):
    """Admin/Teacher model for system access"""
    _tablename_ = 'admins'
    
    id = db.Column(db.String(9), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(AdminRole), default=AdminRole.ADMIN)
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_students = db.relationship('Student', foreign_keys='Student.created_by', backref='creator')
    attendance_logs = db.relationship('AttendanceLog', backref='admin')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role.value,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat()
        }

class Student(db.Model):
    """Student model"""
    _tablename_ = 'students'
    
    id = db.Column(db.String(9), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.Text, nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    parent_phone = db.Column(db.String(20), nullable=True)
    
    # Voice registration status
    voice_registered = db.Column(db.Boolean, default=False)
    voice_registration_date = db.Column(db.DateTime, nullable=True)
    voice_quality_score = db.Column(db.Float, default=0.0)
    
    # Metadata
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.String(9), db.ForeignKey('admins.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    voice_recordings = db.relationship('VoiceData', backref='student', cascade='all, delete-orphan')
    attendances = db.relationship('Attendance', backref='student', cascade='all, delete-orphan')
    
    # Indexes
    _table_args_ = (
        Index('idx_student_class', 'class_name'),
        Index('idx_student_active', 'is_active'),
        Index('idx_student_voice_registered', 'voice_registered'),
    )
    
    def mark_voice_registered(self):
        """Mark student as having completed voice registration"""
        self.voice_registered = True
        self.voice_registration_date = datetime.utcnow()
        db.session.commit()
    
    def get_attendance_rate(self, start_date=None, end_date=None):
        """Calculate attendance rate for a period"""
        query = self.attendances.filter(Attendance.status == AttendanceStatus.HADIR)
        
        if start_date:
            query = query.filter(Attendance.date >= start_date)
        if end_date:
            query = query.filter(Attendance.date <= end_date)
        
        attended_days = query.count()
        
        # Calculate total possible days (this is simplified - you might want to exclude weekends/holidays)
        if start_date and end_date:
            total_days = (end_date - start_date).days + 1
        else:
            total_days = attended_days  # If no date range, return 100% for attended days
        
        return (attended_days / total_days * 100) if total_days > 0 else 0
    
    def to_dict(self, include_voice_data=False):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'class': self.class_name,
            'email': self.email,
            'phone': self.phone,
            'address': self.address,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'parent_phone': self.parent_phone,
            'voice_registered': self.voice_registered,
            'voice_registration_date': self.voice_registration_date.isoformat() if self.voice_registration_date else None,
            'voice_quality_score': self.voice_quality_score,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if include_voice_data:
            data['voice_recordings_count'] = len(self.voice_recordings)
            data['total_attendances'] = len(self.attendances)
        
        return data

class VoiceData(db.Model):
    """Voice biometric data storage"""
    _tablename_ = 'voice_data'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(9), db.ForeignKey('students.id'), nullable=False)
    recording_number = db.Column(db.Integer, nullable=False)  # 1, 2, or 3
    
    # Voice embeddings (stored as JSON)
    resemblyzer_embedding = db.Column(db.Text, nullable=True)  # JSON string
    speechbrain_embedding = db.Column(db.Text, nullable=True)  # JSON string
    
    # Quality metrics
    audio_duration = db.Column(db.Float, nullable=True)
    quality_score = db.Column(db.Float, nullable=True)
    quality_rating = db.Column(db.Enum(VoiceQuality), nullable=True)
    snr_ratio = db.Column(db.Float, nullable=True)
    
    # File information
    original_filename = db.Column(db.String(255), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)
    audio_format = db.Column(db.String(10), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints
    _table_args_ = (
        UniqueConstraint('student_id', 'recording_number', name='uq_student_recording'),
        Index('idx_voice_student', 'student_id'),
        Index('idx_voice_quality', 'quality_score'),
    )
    
    def set_resemblyzer_embedding(self, embedding):
        """Set Resemblyzer embedding"""
        self.resemblyzer_embedding = json.dumps(embedding) if embedding else None
    
    def get_resemblyzer_embedding(self):
        """Get Resemblyzer embedding"""
        return json.loads(self.resemblyzer_embedding) if self.resemblyzer_embedding else None
    
    def set_speechbrain_embedding(self, embedding):
        """Set SpeechBrain embedding"""
        self.speechbrain_embedding = json.dumps(embedding) if embedding else None
    
    def get_speechbrain_embedding(self):
        """Get SpeechBrain embedding"""
        return json.loads(self.speechbrain_embedding) if self.speechbrain_embedding else None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'recording_number': self.recording_number,
            'has_resemblyzer': bool(self.resemblyzer_embedding),
            'has_speechbrain': bool(self.speechbrain_embedding),
            'audio_duration': self.audio_duration,
            'quality_score': self.quality_score,
            'quality_rating': self.quality_rating.value if self.quality_rating else None,
            'snr_ratio': self.snr_ratio,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'audio_format': self.audio_format,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Attendance(db.Model):
    """Daily attendance records"""
    _tablename_ = 'attendances'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(9), db.ForeignKey('students.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.Enum(AttendanceStatus), default=AttendanceStatus.HADIR)
    
    # Voice verification details
    confidence_score = db.Column(db.Float, default=0.0)
    verification_method = db.Column(db.String(50), nullable=True)  # 'voice', 'manual', etc.
    resemblyzer_score = db.Column(db.Float, nullable=True)
    speechbrain_score = db.Column(db.Float, nullable=True)
    
    # Location and device info
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    location_data = db.Column(db.Text, nullable=True)  # JSON string
    
    # Metadata
    notes = db.Column(db.Text, nullable=True)
    verified_by = db.Column(db.String(9), db.ForeignKey('admins.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Constraints and indexes
    _table_args_ = (
        UniqueConstraint('student_id', 'date', name='uq_student_daily_attendance'),
        Index('idx_attendance_date', 'date'),
        Index('idx_attendance_status', 'status'),
        Index('idx_attendance_confidence', 'confidence_score'),
    )
    
    def set_location_data(self, location_dict):
        """Set location data as JSON"""
        self.location_data = json.dumps(location_dict) if location_dict else None
    
    def get_location_data(self):
        """Get location data from JSON"""
        return json.loads(self.location_data) if self.location_data else None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': self.student.name if self.student else None,
            'student_class': self.student.class_name if self.student else None,
            'date': self.date.isoformat(),
            'time': self.time.isoformat(),
            'status': self.status.value,
            'confidence_score': self.confidence_score,
            'verification_method': self.verification_method,
            'resemblyzer_score': self.resemblyzer_score,
            'speechbrain_score': self.speechbrain_score,
            'notes': self.notes,
            'verified_by': self.verified_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class AttendanceLog(db.Model):
    """Audit log for attendance operations"""
    _tablename_ = 'attendance_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendances.id'), nullable=True)
    student_id = db.Column(db.String(9), db.ForeignKey('students.id'), nullable=False)
    admin_id = db.Column(db.String(9), db.ForeignKey('admins.id'), nullable=True)
    
    action = db.Column(db.String(50), nullable=False)  # 'create', 'update', 'delete', 'verify'
    old_values = db.Column(db.Text, nullable=True)  # JSON string
    new_values = db.Column(db.Text, nullable=True)  # JSON string
    reason = db.Column(db.Text, nullable=True)
    
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Indexes
    _table_args_ = (
        Index('idx_log_student', 'student_id'),
        Index('idx_log_admin', 'admin_id'),
        Index('idx_log_action', 'action'),
        Index('idx_log_date', 'created_at'),
    )
    
    def set_old_values(self, values_dict):
        """Set old values as JSON"""
        self.old_values = json.dumps(values_dict) if values_dict else None
    
    def get_old_values(self):
        """Get old values from JSON"""
        return json.loads(self.old_values) if self.old_values else None
    
    def set_new_values(self, values_dict):
        """Set new values as JSON"""
        self.new_values = json.dumps(values_dict) if values_dict else None
    
    def get_new_values(self):
        """Get new values from JSON"""
        return json.loads(self.new_values) if self.new_values else None

class SystemConfig(db.Model):
    """System configuration settings"""
    _tablename_ = 'system_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)
    data_type = db.Column(db.String(20), default='string')  # 'string', 'int', 'float', 'bool', 'json'
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @classmethod
    def get_config(cls, key, default=None):
        """Get configuration value"""
        config = cls.query.filter_by(key=key).first()
        if not config:
            return default
        
        if config.data_type == 'int':
            return int(config.value)
        elif config.data_type == 'float':
            return float(config.value)
        elif config.data_type == 'bool':
            return config.value.lower() in ('true', '1', 'yes')
        elif config.data_type == 'json':
            return json.loads(config.value)
        else:
            return config.value
    
    @classmethod
    def set_config(cls, key, value, data_type='string', description=None, category=None):
        """Set configuration value"""
        config = cls.query.filter_by(key=key).first()
        
        if config:
            config.value = str(value)
            config.data_type = data_type
            if description:
                config.description = description
            if category:
                config.category = category
            config.updated_at = datetime.utcnow()
        else:
            config = cls(
                key=key,
                value=str(value),
                data_type=data_type,
                description=description,
                category=category
            )
            db.session.add(config)
        
        db.session.commit()
        return config

# Database utility functions
def init_db(app):
    """Initialize database with app"""
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        # Create default configurations
        default_configs = [
            ('voice_similarity_threshold', '0.75', 'float', 'Minimum similarity score for voice verification', 'voice'),
            ('max_voice_recordings', '3', 'int', 'Maximum number of voice recordings per student', 'voice'),
            ('attendance_time_window', '08:00-10:00', 'string', 'Time window for attendance marking', 'attendance'),
            ('min_audio_duration', '1.0', 'float', 'Minimum audio duration in seconds', 'voice'),
            ('max_audio_duration', '10.0', 'float', 'Maximum audio duration in seconds', 'voice'),
            ('enable_location_tracking', 'false', 'bool', 'Enable location tracking for attendance', 'security'),
            ('max_daily_attendance_attempts', '5', 'int', 'Maximum attendance attempts per day', 'security'),
        ]
        
        for key, value, data_type, description, category in default_configs:
            if not SystemConfig.query.filter_by(key=key).first():
                SystemConfig.set_config(key, value, data_type, description, category)

def seed_demo_data():
    """Seed database with demo data"""
    try:
        # Create default admin
        if not Admin.query.get('123456789'):
            admin = Admin(
                id='123456789',
                name='Administrator',
                email='admin@smartattendance.com',
                role=AdminRole.SUPER_ADMIN
            )
            admin.set_password('admin123')
            db.session.add(admin)
        
        # Create demo students
        demo_students = [
            {
                'id': '987654321',
                'name': 'Ahmad Fauzi',
                'class_name': 'X-A',
                'email': 'ahmad.fauzi@email.com',
                'phone': '081234567890',
                'created_by': '123456789'
            },
            {
                'id': '876543210',
                'name': 'Siti Nurhaliza',
                'class_name': 'X-B',
                'email': 'siti.nurhaliza@email.com',
                'phone': '081234567891',
                'created_by': '123456789'
            },
            {
                'id': '765432109',
                'name': 'Budi Santoso',
                'class_name': 'XI-A',
                'email': 'budi.santoso@email.com',
                'phone': '081234567892',
                'created_by': '123456789'
            }
        ]
        
        for student_data in demo_students:
            if not Student.query.get(student_data['id']):
                student = Student(**student_data)
                db.session.add(student)
        
        db.session.commit()
        print("Demo data seeded successfully")
        
    except Exception as e:
        print(f"Error seeding demo data: {e}")
        db.session.rollback()

# Database migration utilities
def upgrade_database():
    """Apply database migrations/upgrades"""
    try:
        # Add any database schema upgrades here
        # For example, adding new columns or indexes
        
        # Check if new columns exist and add them if needed
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        
        # Example: Add email column to students if it doesn't exist
        students_columns = [col['name'] for col in inspector.get_columns('students')]
        if 'email' not in students_columns:
            db.engine.execute('ALTER TABLE students ADD COLUMN email VARCHAR(120)')
            print("Added email column to students table")
        
        print("Database upgrade completed")
        
    except Exception as e:
        print(f"Error upgrading database: {e}")

# Database backup and restore utilities
def backup_database(backup_path=None):
    """Create database backup"""
    import sqlite3
    import shutil
    from datetime import datetime
    
    try:
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"backups/smart_attendance_backup_{timestamp}.db"
        
        # Create backup directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # For SQLite databases
        if 'sqlite' in db.engine.url.drivername:
            db_path = db.engine.url.database
            shutil.copy2(db_path, backup_path)
            print(f"Database backed up successfully to {backup_path}")
        else:
            print("Backup method not implemented for this database type")
        
        return backup_path
        
    except Exception as e:
        print(f"Error creating backup: {e}")
        return None

# Database query helpers
class DatabaseHelper:
    """Helper class for common database operations"""
    
    @staticmethod
    def get_student_attendance_summary(student_id, start_date=None, end_date=None):
        """Get attendance summary for a student"""
        query = db.session.query(Attendance).filter_by(student_id=student_id)
        
        if start_date:
            query = query.filter(Attendance.date >= start_date)
        if end_date:
            query = query.filter(Attendance.date <= end_date)
        
        attendances = query.all()
        
        total_days = len(attendances)
        present_days = len([a for a in attendances if a.status == AttendanceStatus.HADIR])
        absent_days = len([a for a in attendances if a.status == AttendanceStatus.ALPHA])
        
        return {
            'total_days': total_days,
            'present_days': present_days,
            'absent_days': absent_days,
            'attendance_rate': (present_days / total_days * 100) if total_days > 0 else 0,
            'latest_attendance': max(attendances, key=lambda x: x.date) if attendances else None
        }
    
    @staticmethod
    def get_class_attendance_summary(class_name, date_filter=None):
        """Get attendance summary for a class"""
        query = db.session.query(Student, Attendance).outerjoin(Attendance)
        query = query.filter(Student.class_name == class_name)
        
        if date_filter:
            query = query.filter(Attendance.date == date_filter)
        
        results = query.all()
        
        students_data = {}
        for student, attendance in results:
            if student.id not in students_data:
                students_data[student.id] = {
                    'student': student,
                    'attendances': []
                }
            if attendance:
                students_data[student.id]['attendances'].append(attendance)
        
        summary = {
            'class_name': class_name,
            'total_students': len(students_data),
            'present_today': 0,
            'absent_today': 0,
            'students': []
        }
        
        for student_id, data in students_data.items():
            student_info = data['student'].to_dict()
            
            if date_filter:
                today_attendance = [a for a in data['attendances'] if a.date == date_filter]
                if today_attendance:
                    student_info['today_status'] = today_attendance[0].status.value
                    if today_attendance[0].status == AttendanceStatus.HADIR:
                        summary['present_today'] += 1
                else:
                    student_info['today_status'] = 'BELUM_ABSEN'
                    summary['absent_today'] += 1
            
            summary['students'].append(student_info)
        
        return summary
    
    @staticmethod
    def get_voice_registration_stats():
        """Get voice registration statistics"""
        total_students = Student.query.count()
        registered_students = Student.query.filter_by(voice_registered=True).count()
        
        # Get quality distribution
        quality_stats = db.session.query(
            VoiceData.quality_rating, 
            db.func.count(VoiceData.id)
        ).group_by(VoiceData.quality_rating).all()
        
        return {
            'total_students': total_students,
            'registered_students': registered_students,
            'registration_rate': (registered_students / total_students * 100) if total_students > 0 else 0,
            'quality_distribution': {rating.value: count for rating, count in quality_stats if rating}
        }
    
    @staticmethod
    def get_attendance_trends(days=30):
        """Get attendance trends for the last N days"""
        from datetime import timedelta
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get daily attendance counts
        daily_counts = db.session.query(
            Attendance.date,
            db.func.count(Attendance.id).label('count')
        ).filter(
            Attendance.date >= start_date,
            Attendance.status == AttendanceStatus.HADIR
        ).group_by(Attendance.date).all()
        
        # Get total students for rate calculation
        total_students = Student.query.filter_by(is_active=True).count()
        
        trends = []
        for attendance_date, count in daily_counts:
            trends.append({
                'date': attendance_date.isoformat(),
                'attendance_count': count,
                'attendance_rate': (count / total_students * 100) if total_students > 0 else 0
            })
        
        return sorted(trends, key=lambda x: x['date'])
    
    @staticmethod
    def cleanup_old_data(days_to_keep=90):
        """Clean up old data to maintain performance"""
        from datetime import timedelta
        
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        
        try:
            # Delete old attendance logs
            old_logs = AttendanceLog.query.filter(AttendanceLog.created_at < cutoff_date).all()
            for log in old_logs:
                db.session.delete(log)
            
            print(f"Cleaned up {len(old_logs)} old attendance logs")
            
            # Optionally archive old attendance records instead of deleting
            # old_attendances = Attendance.query.filter(Attendance.date < cutoff_date).all()
            # Archive logic here...
            
            db.session.commit()
            print("Database cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            db.session.rollback()

# Performance monitoring
class DatabaseMonitor:
    """Monitor database performance and health"""
    
    @staticmethod
    def get_table_sizes():
        """Get size information for all tables"""
        from sqlalchemy import text
        
        try:
            # For SQLite
            if 'sqlite' in db.engine.url.drivername:
                result = db.engine.execute(text("""
                    SELECT name, 
                           (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as row_count
                    FROM sqlite_master m WHERE type='table'
                """))
                
                tables = []
                for row in result:
                    table_name = row[0]
                    # Get actual row count
                    count_result = db.engine.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()
                    
                    tables.append({
                        'table_name': table_name,
                        'row_count': row_count
                    })
                
                return tables
            
        except Exception as e:
            print(f"Error getting table sizes: {e}")
            return []
    
    @staticmethod
    def check_database_health():
        """Perform basic database health checks"""
        health_status = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check connection
            db.engine.execute(text("SELECT 1"))
            
            # Check for orphaned records
            orphaned_voice_data = VoiceData.query.filter(
                ~VoiceData.student_id.in_(db.session.query(Student.id))
            ).count()
            
            if orphaned_voice_data > 0:
                health_status['issues'].append(f"{orphaned_voice_data} orphaned voice data records")
                health_status['recommendations'].append("Run data cleanup to remove orphaned records")
            
            # Check for students without voice registration after 30 days
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            old_unregistered = Student.query.filter(
                Student.voice_registered == False,
                Student.created_at < cutoff_date
            ).count()
            
            if old_unregistered > 0:
                health_status['issues'].append(f"{old_unregistered} students without voice registration after 30 days")
                health_status['recommendations'].append("Follow up on voice registration for these students")
            
            # Check database size (for SQLite)
            if 'sqlite' in db.engine.url.drivername:
                import os
                db_path = db.engine.url.database
                if os.path.exists(db_path):
                    size_mb = os.path.getsize(db_path) / (1024 * 1024)
                    health_status['database_size_mb'] = round(size_mb, 2)
                    
                    if size_mb > 100:  # Warning if DB > 100MB
                        health_status['recommendations'].append("Consider archiving old data or optimizing database")
            
            if health_status['issues']:
                health_status['status'] = 'warning'
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['issues'].append(f"Database connection error: {str(e)}")
        
        return health_status

# Export utilities
def export_data_to_json(output_file=None):
    """Export all data to JSON format"""
    import json
    from datetime import datetime
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"exports/smart_attendance_export_{timestamp}.json"
    
    try:
        # Create export directory
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'students': [s.to_dict(include_voice_data=True) for s in Student.query.all()],
            'admins': [a.to_dict() for a in Admin.query.all()],
            'attendances': [a.to_dict() for a in Attendance.query.all()],
            'voice_data': [v.to_dict() for v in VoiceData.query.all()],
            'system_configs': [
                {
                    'key': c.key,
                    'value': c.value,
                    'data_type': c.data_type,
                    'description': c.description,
                    'category': c.category
                } for c in SystemConfig.query.all()
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Data exported successfully to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        return None