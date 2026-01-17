const API_URL = "https://sentinel-08u6.onrender.com";

// --- Sample Data Config ---
const SAMPLE_DATA = {
    legit: [
        { Time: 10000, Amount: 15.50, V1: -0.5, V2: 0.2, V3: 1.1, V4: -0.5, V5: -0.2, V6: 0.1, V7: 0.2, V8: 0.1, V9: -0.1, V10: 0.1, V11: -0.2, V12: 0.3, V13: 0.1, V14: -0.1, V15: 0.1, V16: -0.2, V17: 0.1, V18: 0.1, V19: -0.1, V20: 0.1, V21: -0.1, V22: 0.2, V23: 0.1, V24: -0.1, V25: -0.1, V26: 0.1, V27: 0.1, V28: 0.1 },
        { Time: 25000, Amount: 99.99, V1: 1.2, V2: -0.3, V3: 0.5, V4: -0.2, V5: -0.4, V6: 0.1, V7: -0.3, V8: 0.0, V9: 0.5, V10: -0.1, V11: -0.5, V12: 0.2, V13: 0.3, V14: -0.2, V15: 0.4, V16: 0.1, V17: -0.3, V18: 0.2, V19: -0.2, V20: -0.1, V21: -0.2, V22: 0.1, V23: 0.0, V24: 0.1, V25: 0.2, V26: -0.2, V27: 0.0, V28: 0.0 },
        { Time: 45000, Amount: 5.00, V1: -0.8, V2: 0.6, V3: 0.9, V4: -0.6, V5: 0.1, V6: -0.2, V7: 0.3, V8: 0.2, V9: -0.2, V10: 0.2, V11: -0.1, V12: 0.1, V13: 0.0, V14: -0.1, V15: 0.2, V16: -0.1, V17: 0.1, V18: 0.0, V19: -0.1, V20: 0.1, V21: 0.1, V22: 0.1, V23: -0.1, V24: 0.0, V25: -0.1, V26: 0.1, V27: 0.1, V28: 0.0 },
        { Time: 80000, Amount: 1200.00, V1: 0.5, V2: -0.8, V3: 0.2, V4: 0.1, V5: -0.5, V6: -0.1, V7: -0.2, V8: -0.1, V9: 0.1, V10: -0.2, V11: 0.1, V12: -0.1, V13: 0.2, V14: -0.1, V15: -0.2, V16: 0.1, V17: -0.1, V18: 0.0, V19: 0.1, V20: 0.2, V21: 0.0, V22: -0.1, V23: 0.1, V24: -0.2, V25: 0.0, V26: -0.1, V27: 0.0, V28: 0.0 },
        { Time: 120000, Amount: 24.95, V1: -1.5, V2: 1.2, V3: 0.1, V4: -1.0, V5: 0.5, V6: -0.5, V7: 0.4, V8: 0.3, V9: -0.4, V10: 0.3, V11: -0.3, V12: 0.2, V13: 0.0, V14: -0.2, V15: 0.1, V16: -0.1, V17: 0.2, V18: 0.1, V19: -0.2, V20: 0.1, V21: 0.1, V22: 0.2, V23: -0.2, V24: 0.1, V25: -0.1, V26: 0.2, V27: 0.1, V28: 0.1 }
    ],
    fraud: [
        { Time: 406, Amount: 0.0, V1: -2.31, V2: 1.95, V3: -1.6, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39, V9: -2.77, V10: -2.77, V11: 3.2, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38, V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03, V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14 },
        { Time: 472, Amount: 529.0, V1: -3.04, V2: -3.15, V3: 1.08, V4: 2.28, V5: 1.35, V6: -1.06, V7: 0.32, V8: -0.06, V9: -0.26, V10: -0.4, V11: -0.05, V12: -0.62, V13: 0.72, V14: -2.48, V15: 2.15, V16: 1.15, V17: 1.99, V18: 0.55, V19: 0.32, V20: 2.1, V21: 0.66, V22: 0.43, V23: 1.37, V24: -0.29, V25: 0.27, V26: -0.14, V27: -0.25, V28: 0.03 },
        { Time: 4462, Amount: 239.93, V1: -2.3, V2: 1.28, V3: -0.31, V4: 0.96, V5: -0.9, V6: -0.59, V7: -0.93, V8: 0.85, V9: -0.11, V10: -1.63, V11: 1.83, V12: -2.31, V13: -1.22, V14: -2.63, V15: 0.23, V16: -2.25, V17: -3.53, V18: -0.91, V19: -0.28, V20: -0.25, V21: 0.21, V22: -0.25, V23: 0.07, V24: -0.13, V25: -0.22, V26: -0.55, V27: 0.5, V28: 0.37 },
        { Time: 6986, Amount: 1809.68, V1: -4.39, V2: 1.35, V3: -2.59, V4: 2.6, V5: -1.18, V6: -0.24, V7: -1.35, V8: 1.15, V9: -2.71, V10: -4.89, V11: 3.58, V12: -6.85, V13: -1.2, V14: -6.74, V15: 0.69, V16: -4.43, V17: -7.56, V18: -2.73, V19: 0.85, V20: -0.17, V21: 0.57, V22: 0.17, V23: -0.43, V24: -0.05, V25: 0.25, V26: -0.01, V27: -0.04, V28: 0.72 },
        { Time: 7519, Amount: 1.0, V1: 1.23, V2: 3.01, V3: -4.3, V4: 4.73, V5: 3.62, V6: -1.35, V7: 1.71, V8: -0.47, V9: -2.58, V10: -2.29, V11: 2.38, V12: -2.73, V13: -0.9, V14: -5.79, V15: -0.91, V16: -0.9, V17: 2.24, V18: 0.3, V19: -2.91, V20: 0.0, V21: -0.37, V22: -0.7, V23: -0.51, V24: -0.09, V25: 1.29, V26: 0.43, V27: 0.02, V28: 0.22 }
    ]
};

// --- DOM Elements ---
const form = document.getElementById('transaction-form');
const featuresContainer = document.getElementById('features-container');
const inputTime = document.getElementById('input-time');
const inputAmount = document.getElementById('input-amount');
const btnRandomLegit = document.getElementById('btn-random-legit');
const btnRandomFraud = document.getElementById('btn-random-fraud');

// Simulator & Results
const simulator = document.getElementById('simulator');
const simSteps = [document.getElementById('step-req'), document.getElementById('step-proc'), document.getElementById('step-res')];
const simPayload = document.getElementById('simulator-payload');
const resultDisplay = document.getElementById('result-display');
const emptyState = document.getElementById('empty-state');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    generateFeatureInputs();
    loadSample(SAMPLE_DATA.legit[0]); // Load initial default
});

// --- Functions ---

function generateFeatureInputs() {
    featuresContainer.innerHTML = '';
    for (let i = 1; i <= 28; i++) {
        const div = document.createElement('div');
        div.className = 'feature-input';
        div.innerHTML = `
            <input type="number" step="0.01" id="v${i}" placeholder="V${i}" title="V${i}">
        `;
        featuresContainer.appendChild(div);
    }
}

function loadSample(data) {
    inputTime.value = data.Time;
    inputAmount.value = data.Amount;
    for (let i = 1; i <= 28; i++) {
        const el = document.getElementById(`v${i}`);
        if (el) el.value = data[`V${i}`];
    }
}

function getRandom(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

// --- Event Listeners ---

btnRandomLegit.addEventListener('click', (e) => {
    e.preventDefault();
    loadSample(getRandom(SAMPLE_DATA.legit));
});

btnRandomFraud.addEventListener('click', (e) => {
    e.preventDefault();
    loadSample(getRandom(SAMPLE_DATA.fraud));
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // 1. Build Payload
    const payload = {
        Time: parseFloat(inputTime.value) || 0,
        Amount: parseFloat(inputAmount.value) || 0
    };
    for (let i = 1; i <= 28; i++) {
        const val = document.getElementById(`v${i}`).value;
        payload[`V${i}`] = parseFloat(val) || 0;
    }

    // 2. Start Simulation Visuals
    resultDisplay.classList.add('hidden');
    emptyState.classList.add('hidden');
    simulator.classList.remove('hidden');

    simPayload.textContent = JSON.stringify(payload, null, 2);

    updateSimStep(0); // Sending

    try {
        // 3. API Request
        updateSimStep(1); // Processing (fake delay for effect)

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('API Error');
        const result = await response.json();

        // 4. Show Result
        setTimeout(() => {
            updateSimStep(2); // Receive
            setTimeout(() => {
                simulator.classList.add('hidden');
                displayResult(result);
            }, 500);
        }, 800);

    } catch (error) {
        console.error(error);
        alert('Connection Failed');
        simulator.classList.add('hidden');
        emptyState.classList.remove('hidden');
    }
});

function updateSimStep(idx) {
    simSteps.forEach((el, i) => {
        el.className = i === idx ? 'step active' : 'step';
    });
}

function displayResult(res) {
    resultDisplay.classList.remove('hidden');

    const badge = document.getElementById('result-badge');
    const prob = document.getElementById('res-prob');
    const pred = document.getElementById('res-pred');

    if (res.is_fraud) {
        badge.className = 'result-badge badge-fraud';
        badge.innerHTML = `
            <i data-lucide="alert-triangle" style="width: 48px; height: 48px;"></i>
            <h3>FRAUD DETECTED</h3>
        `;
        prob.style.color = 'var(--danger)';
    } else {
        badge.className = 'result-badge badge-legit';
        badge.innerHTML = `
            <i data-lucide="shield-check" style="width: 48px; height: 48px;"></i>
            <h3>LEGITIMATE TRANSACTION</h3>
        `;
        prob.style.color = 'var(--success)';
    }

    prob.textContent = (res.fraud_probability * 100).toFixed(2) + "%";
    pred.textContent = res.prediction;

    lucide.createIcons();
}
