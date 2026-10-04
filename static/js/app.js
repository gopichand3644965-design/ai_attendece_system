document.addEventListener("DOMContentLoaded", () => {
    // ── API Configuration ───────────────────────────────────
    const REQUEST_TIMEOUT_MS = 60000; // 60s timeout for Render cold starts
    const MAX_RETRIES = 2;
    const RETRY_DELAY_MS = 4000; // 4s between retries

    // Elements
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("snapshot-canvas");
    const startCameraBtn = document.getElementById("start-camera-btn");
    const serverStatus = document.getElementById("server-status");
    
    // Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    // Attendance
    const startMonitorBtn = document.getElementById("start-monitor-btn");
    const stopMonitorBtn = document.getElementById("stop-monitor-btn");
    const scanningOverlay = document.getElementById("scanning-overlay");
    const attendanceResult = document.getElementById("attendance-result");
    
    // Registration
    const captureRegisterBtn = document.getElementById("capture-register-btn");
    const registerForm = document.getElementById("register-form");
    const studentIdInput = document.getElementById("student-id");
    const studentNameInput = document.getElementById("student-name");
    const registerResult = document.getElementById("register-result");
    
    // Settings
    const saveSettingsBtn = document.getElementById("save-settings-btn");
    const startTimeInput = document.getElementById("start-time");
    const endTimeInput = document.getElementById("end-time");
    const settingsResult = document.getElementById("settings-result");

    let stream = null;
    let monitorInterval = null;
    let isMonitoring = false;
    let isProcessingFrame = false;
    let appSettings = { attendance_start_time: "08:00", attendance_end_time: "15:00" };

    // ── Server Status Banner ────────────────────────────────

    function showServerStatus(message, isError = false) {
        if (!serverStatus) return;
        serverStatus.querySelector("span").textContent = message;
        serverStatus.classList.remove("hidden", "error");
        if (isError) serverStatus.classList.add("error");
    }

    function hideServerStatus() {
        if (serverStatus) serverStatus.classList.add("hidden");
    }

    // ── API Request Wrapper with Timeout & Retry ────────────
    // Handles Render cold starts (30-60s wake-up) gracefully.

    async function apiRequest(url, options = {}, retries = MAX_RETRIES) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
            });
            clearTimeout(timeoutId);
            hideServerStatus();
            return response;
        } catch (err) {
            clearTimeout(timeoutId);

            if (retries > 0) {
                const attempt = MAX_RETRIES - retries + 1;
                showServerStatus(
                    `Server is waking up\u2026 retrying (${attempt}/${MAX_RETRIES})`
                );
                await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
                return apiRequest(url, options, retries - 1);
            }

            showServerStatus("Could not reach the server. Please try again.", true);
            throw err;
        }
    }

    // ── Load Settings ───────────────────────────────────────
    async function fetchSettings() {
        try {
            const res = await apiRequest("/settings/");
            appSettings = await res.json();
            startTimeInput.value = appSettings.attendance_start_time;
            endTimeInput.value = appSettings.attendance_end_time;
        } catch (e) {
            console.error("Failed to load settings:", e);
        }
    }
    fetchSettings();

    // ── Tabs Logic ──────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(btn.dataset.target).classList.add("active");
        });
    });

    // ── Camera Logic ────────────────────────────────────────
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, facingMode: "user" } 
            });
            video.srcObject = stream;
            
            startCameraBtn.textContent = "Camera Active";
            startCameraBtn.classList.replace("primary-btn", "success-btn");
            startCameraBtn.disabled = true;

            // Enable action buttons
            startMonitorBtn.disabled = false;
            captureRegisterBtn.disabled = false;
            
            attendanceResult.innerHTML = "Camera ready. Click Start Monitoring.";
            registerResult.innerHTML = "Camera ready. Fill details and capture.";
            
        } catch (err) {
            console.error("Camera error:", err);
            alert("Failed to access webcam. Please allow permissions.");
        }
    }

    startCameraBtn.addEventListener("click", startCamera);

    // Helper: Capture Frame to Blob
    function captureFrame() {
        return new Promise((resolve) => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            
            // Mirror image to match video element
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob((blob) => {
                resolve(blob);
            }, "image/jpeg", 0.9);
        });
    }

    // ── Live Attendance Logic ───────────────────────────────
    
    async function processAttendanceFrame() {
        if (!isMonitoring || isProcessingFrame) return;
        isProcessingFrame = true;
        
        try {
            const blob = await captureFrame();
            
            const formData = new FormData();
            formData.append("file", blob, "live_frame.jpg");

            // No retries for monitoring frames — the next interval will try again
            const response = await apiRequest("/attendance/recognize", {
                method: "POST",
                body: formData
            }, 0);

            const data = await response.json();

            if (data.success && data.recognized) {
                attendanceResult.className = "result-box success";
                attendanceResult.innerHTML = `
                    <strong>\u2705 ${data.name}</strong><br>
                    ID: ${data.student_id}<br>
                    Similarity: ${(data.similarity * 100).toFixed(1)}%<br>
                    <em>Attendance marked for today!</em>
                `;
            } else if (data.success && !data.recognized) {
                attendanceResult.className = "result-box error";
                attendanceResult.innerHTML = `
                    <strong>\u2753 Unknown Face Detected</strong><br>
                    Please register this student first.
                `;
            } else {
                // No face detected, ignore silently or show small dot
            }
        } catch (err) {
            console.error("Error recognizing face:", err);
        } finally {
            isProcessingFrame = false;
        }
    }

    startMonitorBtn.addEventListener("click", () => {

        isMonitoring = true;
        startMonitorBtn.classList.add("hidden");
        stopMonitorBtn.classList.remove("hidden");
        scanningOverlay.classList.remove("hidden");
        
        attendanceResult.className = "result-box";
        attendanceResult.innerHTML = "Scanning for faces...";
        
        // Process a frame every 1.5 seconds
        monitorInterval = setInterval(processAttendanceFrame, 1500);
    });

    stopMonitorBtn.addEventListener("click", () => {
        isMonitoring = false;
        clearInterval(monitorInterval);
        
        stopMonitorBtn.classList.add("hidden");
        startMonitorBtn.classList.remove("hidden");
        scanningOverlay.classList.add("hidden");
        
        attendanceResult.className = "result-box";
        attendanceResult.innerHTML = "Monitoring stopped.";
    });


    // ── Registration Logic ──────────────────────────────────

    captureRegisterBtn.addEventListener("click", async () => {
        const studentId = studentIdInput.value.trim();
        const studentName = studentNameInput.value.trim();

        if (!studentId || !studentName) {
            alert("Please enter Student ID and Name");
            return;
        }

        registerResult.className = "result-box";
        registerResult.innerHTML = "Capturing face and registering... please wait.";
        captureRegisterBtn.disabled = true;

        try {
            // Capture a frame
            const blob = await captureFrame();

            const formData = new FormData();
            formData.append("student_id", studentId);
            formData.append("name", studentName);
            // Append the single frame as the 'files' array for the backend
            formData.append("files", blob, "register_face.jpg");

            const response = await apiRequest("/students/register", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                registerResult.className = "result-box success";
                registerResult.innerHTML = `
                    <strong>\u2705 Registration Successful!</strong><br>
                    ${studentName} (ID: ${studentId}) is now registered.<br>
                    Extracted ${data.embeddings_count} face embeddings.
                `;
                studentIdInput.value = "";
                studentNameInput.value = "";
            } else {
                registerResult.className = "result-box error";
                registerResult.innerHTML = `
                    <strong>\u274C Registration Failed</strong><br>
                    ${data.message || data.detail}
                `;
            }
        } catch (err) {
            console.error(err);
            registerResult.className = "result-box error";
            registerResult.innerHTML = "Network error. The server may still be starting \u2014 please try again in a moment.";
        } finally {
            captureRegisterBtn.disabled = false;
        }
    });

    // ── Save Settings Logic ─────────────────────────────────
    saveSettingsBtn.addEventListener("click", async () => {
        const start = startTimeInput.value;
        const end = endTimeInput.value;
        
        if (!start || !end) return;

        saveSettingsBtn.disabled = true;
        settingsResult.className = "result-box";
        settingsResult.innerHTML = "Saving...";

        try {
            const res = await apiRequest("/settings/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    attendance_start_time: start,
                    attendance_end_time: end
                })
            });
            const data = await res.json();
            
            if (data.success) {
                appSettings.attendance_start_time = start;
                appSettings.attendance_end_time = end;
                settingsResult.className = "result-box success";
                settingsResult.innerHTML = "\u2705 Settings saved successfully! Auto-absent job scheduled.";
            } else {
                settingsResult.className = "result-box error";
                settingsResult.innerHTML = "\u274C Failed to save settings.";
            }
        } catch (e) {
            settingsResult.className = "result-box error";
            settingsResult.innerHTML = "\u274C Network error. The server may still be starting \u2014 please try again.";
        } finally {
            saveSettingsBtn.disabled = false;
        }
    });
});
