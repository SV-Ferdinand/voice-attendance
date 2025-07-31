// Data storage (simulasi database)
let students = [];
let attendanceRecords = [];
let voiceRecordings = {};
let currentAdmin = null;
let recordingStep = 0;
let voiceRegistrationCount = 0;

// Fungsi navigasi halaman
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId).classList.add('active');
}

// Fungsi untuk menampilkan step 2
function showStep2() {
    document.getElementById('attendance-step-1').classList.add('hidden');
    document.getElementById('attendance-step-2').classList.remove('hidden');
}

// Fungsi untuk menampilkan step 3
function showStep3() {
    const studentId = document.getElementById('studentId').value;
    const studentName = document.getElementById('studentName').value;
    
    if (!studentId || studentId.length !== 9 || !studentName) {
        alert('Mohon lengkapi data dengan benar (ID harus 9 digit)');
        return;
    }
    
    document.getElementById('attendance-step-2').classList.add('hidden');
    document.getElementById('attendance-step-3').classList.remove('hidden');
}

// Simulasi perekaman suara
function startRecording(step) {
    const recordBtn = document.getElementById(`recordBtn${step}`);
    const status = document.getElementById(`status${step}`);
    const visualizer = document.getElementById(`visualizer${step}`);
    
    recordBtn.textContent = '🔴 Merekam...';
    recordBtn.classList.add('recording');
    recordBtn.disabled = true;
    
    visualizer.textContent = 'Sedang merekam... Ucapkan "HADIR"';
    status.className = 'status-message status-info';
    status.textContent = 'Mendengarkan suara Anda...';
    status.classList.remove('hidden');
    
    // Simulasi proses perekaman
    setTimeout(() => {
        recordBtn.textContent = '✅ Rekaman Selesai';
        recordBtn.classList.remove('recording');
        recordBtn.disabled = false;
        
        visualizer.textContent = 'Rekaman berhasil disimpan';
        status.className = 'status-message status-success';
        status.textContent = 'Suara berhasil direkam!';
        
        if (step === 1) {
            document.getElementById('nextStep1').classList.remove('hidden');
        } else if (step === 2) {
            document.getElementById('verifyBtn').classList.remove('hidden');
        }
    }, 3000);
}

// Verifikasi absensi
function verifyAttendance() {
    const studentId = document.getElementById('studentId').value;
    const studentName = document.getElementById('studentName').value;
    
    // Simulasi verifikasi suara
    setTimeout(() => {
        // Update hasil absensi
        document.getElementById('resultId').textContent = studentId;
        document.getElementById('resultName').textContent = studentName;
        document.getElementById('resultDate').textContent = new Date().toLocaleDateString('id-ID');
        
        // Simpan ke records
        attendanceRecords.push({
            id: studentId,
            name: studentName,
            class: '-',
            date: new Date().toLocaleDateString('id-ID'),
            time: new Date().toLocaleTimeString('id-ID')
        });
        
        // Tampilkan hasil
        document.getElementById('attendance-step-3').classList.add('hidden');
        document.getElementById('attendance-result').classList.remove('hidden');
    }, 2000);
}

// Reset form absensi
function resetAttendance() {
    document.getElementById('attendance-result').classList.add('hidden');
    document.getElementById('attendance-step-1').classList.remove('hidden');
    document.getElementById('attendance-step-2').classList.add('hidden');
    document.getElementById('attendance-step-3').classList.add('hidden');
    document.getElementById('nextStep1').classList.add('hidden');
    document.getElementById('verifyBtn').classList.add('hidden');
    
    // Reset form
    document.getElementById('studentId').value = '';
    document.getElementById('studentName').value = '';
    document.getElementById('status1').classList.add('hidden');
    document.getElementById('status2').classList.add('hidden');
    document.getElementById('recordBtn1').textContent = '🎤 Mulai Rekam';
    document.getElementById('recordBtn2').textContent = '🎤 Verifikasi Suara';
}

// Login admin
function adminLogin() {
    const adminId = document.getElementById('adminId').value;
    const adminName = document.getElementById('adminName').value;
    const status = document.getElementById('adminLoginStatus');
    
    if (!adminId || adminId.length !== 9 || !adminName) {
        status.className = 'status-message status-error';
        status.textContent = 'Mohon lengkapi data dengan benar (ID harus 9 digit)';
        status.classList.remove('hidden');
        return;
    }
    
    // Simulasi login berhasil
    currentAdmin = { id: adminId, name: adminName };
    document.getElementById('adminNameDisplay').textContent = adminName;
    showPage('admin-dashboard');
    
    // Reset form
    document.getElementById('adminId').value = '';
    document.getElementById('adminName').value = '';
    status.classList.add('hidden');
    
    updateStudentSelect();
    updateTables();
}

// Registrasi user baru
function registerUser() {
    const studentId = document.getElementById('newStudentId').value;
    const studentName = document.getElementById('newStudentName').value;
    const studentClass = document.getElementById('newStudentClass').value;
    const status = document.getElementById('registrationStatus');
    
    if (!studentId || studentId.length !== 9 || !studentName || !studentClass) {
        status.className = 'status-message status-error';
        status.textContent = 'Mohon lengkapi semua data dengan benar (ID harus 9 digit)';
        status.classList.remove('hidden');
        return;
    }
    
    // Cek duplikasi ID
    if (students.some(s => s.id === studentId)) {
        status.className = 'status-message status-error';
        status.textContent = 'ID siswa sudah terdaftar!';
        status.classList.remove('hidden');
        return;
    }
    
    // Tambah siswa baru
    students.push({
        id: studentId,
        name: studentName,
        class: studentClass,
        voiceRegistered: false
    });
    
    status.className = 'status-message status-success';
    status.textContent = 'Siswa berhasil didaftarkan!';
    status.classList.remove('hidden');
    
    // Reset form
    document.getElementById('newStudentId').value = '';
    document.getElementById('newStudentName').value = '';
    document.getElementById('newStudentClass').value = '';
    
    updateStudentSelect();
    updateTables();
}

// Update dropdown siswa
function updateStudentSelect() {
    const select = document.getElementById('voiceStudentId');
    select.innerHTML = '<option value="">Pilih siswa...</option>';
    
    students.forEach(student => {
        const option = document.createElement('option');
        option.value = student.id;
        option.textContent = `${student.id} - ${student.name}`;
        select.appendChild(option);
    });
    
    select.addEventListener('change', function() {
        const recordBtn = document.getElementById('recordBtn3');
        const visualizer = document.getElementById('visualizer3');
        
        if (this.value) {
            recordBtn.disabled = false;
            visualizer.textContent = 'Siap untuk merekam suara...';
        } else {
            recordBtn.disabled = true;
            visualizer.textContent = 'Pilih siswa terlebih dahulu...';
        }
    });
}

// Mulai registrasi suara
function startVoiceRegistration() {
    const studentId = document.getElementById('voiceStudentId').value;
    if (!studentId) return;
    
    voiceRegistrationCount++;
    const recordBtn = document.getElementById('recordBtn3');
    const status = document.getElementById('voiceRegStatus');
    const visualizer = document.getElementById('visualizer3');
    
    recordBtn.textContent = '🔴 Merekam...';
    recordBtn.classList.add('recording');
    recordBtn.disabled = true;
    
    visualizer.textContent = `Merekam sampel ${voiceRegistrationCount}/3...`;
    status.className = 'status-message status-info';
    status.textContent = `Merekam sampel suara ${voiceRegistrationCount} dari 3...`;
    status.classList.remove('hidden');
    
    setTimeout(() => {
        recordBtn.classList.remove('recording');
        recordBtn.disabled = false;
        
        if (voiceRegistrationCount < 3) {
            recordBtn.textContent = `🎤 Rekam Lagi (${voiceRegistrationCount + 1}/3)`;
            visualizer.textContent = `Sampel ${voiceRegistrationCount} selesai. Siap untuk sampel berikutnya...`;
            status.className = 'status-message status-success';
            status.textContent = `Sampel ${voiceRegistrationCount} berhasil direkam!`;
        } else {
            recordBtn.textContent = '✅ Registrasi Selesai';
            recordBtn.disabled = true;
            visualizer.textContent = 'Semua sampel suara berhasil direkam!';
            status.className = 'status-message status-success';
            status.textContent = 'Registrasi suara berhasil! Data biometrik telah disimpan.';
            
            // Update status siswa
            const student = students.find(s => s.id === studentId);
            if (student) {
                student.voiceRegistered = true;
                voiceRecordings[studentId] = true; // Simulasi data suara tersimpan
            }
            
            voiceRegistrationCount = 0;
            updateTables();
        }
    }, 3000);
}

// Update tabel data
function updateTables() {
    // Update tabel siswa
    const studentTableBody = document.getElementById('studentTableBody');
    if (students.length === 0) {
        studentTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #666;">Belum ada data siswa</td></tr>';
    } else {
        studentTableBody.innerHTML = students.map(student => `
            <tr>
                <td>${student.id}</td>
                <td>${student.name}</td>
                <td>${student.class}</td>
                <td>${student.voiceRegistered ? '✅ Sudah' : '❌ Belum'}</td>
            </tr>
        `).join('');
    }
    
    // Update tabel absensi
    const attendanceTableBody = document.getElementById('attendanceTableBody');
    const todayRecords = attendanceRecords.filter(record => 
        record.date === new Date().toLocaleDateString('id-ID')
    );
    
    if (todayRecords.length === 0) {
        attendanceTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #666;">Belum ada data absensi hari ini</td></tr>';
    } else {
        attendanceTableBody.innerHTML = todayRecords.map(record => `
            <tr>
                <td>${record.id}</td>
                <td>${record.name}</td>
                <td>${record.class}</td>
                <td>${record.date}</td>
                <td>${record.time}</td>
            </tr>
        `).join('');
    }
}

// Inisialisasi saat halaman dimuat
document.addEventListener('DOMContentLoaded', function() {
    updateTables();
});