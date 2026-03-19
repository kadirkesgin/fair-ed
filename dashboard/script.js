const dict = {
    EN: {
        appTitle: "Fair-Ed <span>Dashboard</span>",
        appSubtitle: "Causal AI Decision Support for Equitable Educational Interventions",
        selectTitle: '<i class="fa-solid fa-user-graduate"></i> Select or Enter At-Risk Student',
        optDefault: "Select a scenario or enter custom student...",
        opt1: "Student #1042 (Low SES, Frequent Absences)",
        opt2: "Student #215 (High SES, Low Weekly Study Time)",
        optCustom: "➕ Enter Custom Student Data",
        customFormTitle: "Enter Student Metrics",
        lblAbsences: "Absences (Days)",
        lblStudy: "Study Time (Hrs/Wk)",
        lblFree: "Free Time (1-5)",
        lblSES: "Socioeconomic Status (SES)",
        sesLow: "Low SES",
        sesMid: "Middle SES",
        sesHigh: "High SES",
        currentObj: "Current Academic State",
        statAbs: "Absences",
        statStu: "Study Time",
        statFre: "Free Time",
        curPredTitle: "Current Prediction: Failure Risk",
        probPassText: "Probability of Passing:",
        btnRun: "Generate Fair Recourse (DiCE)",
        loadingText: "Solving Causal Optimization for Fair Recourse...",
        successTitle: "Fair Pedagogical Intervention Recommended",
        postInterv: "Post-Intervention Probability:",
        actionPlanTitle: "Actionable Pathway for Student",
        actionPlanDesc: "The AI has identified the mathematically optimal adjustments tailored to this student's socioeconomic capabilities to flip their prediction to 'Pass'.",
        metricCost: "Intervention Cost",
        metricRFD: "Recourse Fairness Diff. (RFD)",
        metricOpt: "(Optimal)",
        btnReset: "Analyze Another Student",
        days: "Days",
        hrWk: "hr/wk",
        freeVal: "/5",
        lowSesText: "Low SES",
        midSesText: "Middle SES",
        highSesText: "High SES"
    },
    TR: {
        appTitle: "Fair-Ed <span>Öğretmen Paneli</span>",
        appSubtitle: "Adil Eğitim Müdahaleleri İçin Nedensel YZ Karar Destek Sistemi",
        selectTitle: '<i class="fa-solid fa-user-graduate"></i> Riskli Öğrenciyi Seçin veya Girin',
        optDefault: "Listeden seçin veya manuel girin...",
        opt1: "Öğrenci #1042 (Düşük SES, Sık Devamsızlık)",
        opt2: "Öğrenci #215 (Yüksek SES, Düşük Çalışma Süresi)",
        optCustom: "➕ Özel Öğrenci Verisi Gir",
        customFormTitle: "Öğrenci Metriklerini Girin",
        lblAbsences: "Devamsızlık (Gün)",
        lblStudy: "Çalışma (Saat/Hafta)",
        lblFree: "Boş Zaman (1-5)",
        lblSES: "Sosyoekonomik Durum (SES)",
        sesLow: "Düşük SES",
        sesMid: "Orta SES",
        sesHigh: "Yüksek SES",
        currentObj: "Mevcut Akademik Durum",
        statAbs: "Devamsızlık",
        statStu: "Çalışma",
        statFre: "Boş Zaman",
        curPredTitle: "Mevcut Tahmin: Kalma Riski",
        probPassText: "Geçme İhtimali:",
        btnRun: "Adil Telafi (Recourse) Üret",
        loadingText: "Adil Müdahale için SCM Optimizasyonu Çözülüyor...",
        successTitle: "Adil Pedagojik Müdahale Önerisi",
        postInterv: "Müdahale Sonrası İhtimal:",
        actionPlanTitle: "Öğrenci İçin Eylem Planı",
        actionPlanDesc: "Yapay zeka, bu öğrencinin geçebilmesi için sosyoekonomik kapasitesine en uygun ve en adil (düşük maliyetli) aksiyonları belirlemiştir.",
        metricCost: "Müdahale Maliyeti",
        metricRFD: "Telafi Adaletsizlik Farkı (RFD)",
        metricOpt: "(Optimum)",
        btnReset: "Başka Öğrenci Analiz Et",
        days: "Gün",
        hrWk: "saat/hft",
        freeVal: "/5",
        lowSesText: "Düşük SES",
        midSesText: "Orta SES",
        highSesText: "Yüksek SES"
    }
};

let currentLang = 'EN';

function applyLanguage(lang) {
    currentLang = lang;
    document.getElementById("lang-en").className = lang === 'EN' ? 'active' : '';
    document.getElementById("lang-tr").className = lang === 'TR' ? 'active' : '';

    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if(dict[lang][key]) el.innerHTML = dict[lang][key];
    });
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("lang-en").addEventListener("click", () => applyLanguage("EN"));
    document.getElementById("lang-tr").addEventListener("click", () => applyLanguage("TR"));

    const studentSelect = document.getElementById("student-select");
    const customForm = document.getElementById("custom-form");
    const studentProfile = document.getElementById("student-profile");
    const runBtn = document.getElementById("run-ai-btn");
    const processingPanel = document.getElementById("ai-processing");
    const resultPanel = document.getElementById("recourse-result");
    const resetBtn = document.getElementById("reset-btn");
    
    let isCustom = false;

    // Fixed pre-processed students for UI demo
    const fixedData = {
        stu1: { abs: 18, stu: 1, fre: 4, ses: "Low SES", prob: "25%" },
        stu2: { abs: 4, stu: 0.5, fre: 5, ses: "High SES", prob: "41%" }
    };

    function refreshProfileStats(data) {
        document.getElementById("cur-absences").innerText = `${data.abs} ${dict[currentLang].days}`;
        document.getElementById("cur-study").innerText = `${data.stu} ${dict[currentLang].hrWk}`;
        document.getElementById("cur-free").innerText = `${data.fre} ${dict[currentLang].freeVal}`;
        
        // Handle SES translation and color class mapping (Low/Mid/High)
        let sesStr = "";
        let sesClass = "";
        if (data.ses.includes("Low")) {
            sesStr = dict[currentLang].lowSesText;
            sesClass = "tag-low";
        } else if (data.ses.includes("Mid")) {
            sesStr = dict[currentLang].midSesText;
            sesClass = "tag-mid";
        } else {
            sesStr = dict[currentLang].highSesText;
            sesClass = "tag-high";
        }
        
        document.getElementById("cur-ses").innerText = sesStr;
        document.getElementById("cur-ses").className = "stat-value " + sesClass;
        document.getElementById("pass-prob").innerText = data.prob;
    }

    function updateProfileView(data) {
        refreshProfileStats(data);
        customForm.classList.add("hidden");
        studentProfile.classList.remove("hidden");
        resultPanel.classList.add("hidden");
    }

    studentSelect.addEventListener("change", (e) => {
        const val = e.target.value;
        if(val === "custom") {
            isCustom = true;
            customForm.classList.remove("hidden");
            studentProfile.classList.remove("hidden");
            resultPanel.classList.add("hidden");
            updateCustomProfile();
        } else if(fixedData[val]) {
            isCustom = false;
            updateProfileView(fixedData[val]);
        }
    });

    function updateCustomProfile() {
        if(!isCustom) return;
        const abs = document.getElementById("inp-absences").value || 0;
        const stu = document.getElementById("inp-study").value || 0;
        const fre = document.getElementById("inp-free").value || 3;
        const ses = document.getElementById("inp-ses").value; // e.g. "Low SES", "Middle SES"
        
        // Mock Predictive Probability Logic for the demo
        let penalty = (abs * 2) - (stu * 15);
        let probNum = 80 - penalty;
        if(probNum < 5) probNum = 5;
        if(probNum > 95) probNum = 95;

        refreshProfileStats({
            abs: abs, stu: stu, fre: fre, ses: ses, prob: `${Math.round(probNum)}%`
        });
    }

    // Live update profile as user types
    document.querySelectorAll(".form-grid input, .form-grid select").forEach(el => {
        el.addEventListener("input", updateCustomProfile);
    });

    runBtn.addEventListener("click", () => {
        studentProfile.classList.add("hidden");
        customForm.classList.add("hidden");
        processingPanel.classList.remove("hidden");
        
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if(progress > 100) progress = 100;
            document.querySelector(".progress-bar").style.width = `${progress}%`;
            
            if(progress === 100) {
                clearInterval(interval);
                setTimeout(showResults, 500);
            }
        }, 200);
    });

    function showResults() {
        processingPanel.classList.add("hidden");
        resultPanel.classList.remove("hidden");
        
        const actionList = document.getElementById("action-list");
        actionList.innerHTML = '';
        
        let sesStr = document.getElementById("cur-ses").innerText;
        let isLowSES = sesStr.includes(dict[currentLang].lowSesText);
        let isMidSES = sesStr.includes(dict[currentLang].midSesText);
        let abs = parseInt(document.getElementById("cur-absences").innerText);
        
        // Adaptive phrasing based on exact SES tier
        let constraintEn = isLowSES ? "<em>(Max cost-effective constraint applied)</em>" : 
                           (isMidSES ? "<em>(Moderate cost constraint applied)</em>" : "");
                           
        let constraintTr = isLowSES ? "<em>(Sıkı adil maliyet filtresi uygulandı)</em>" : 
                           (isMidSES ? "<em>(Orta seviye kapasite filtresi uygulandı)</em>" : "");
        
        let act1 = currentLang === 'EN' ? 
            `Increase weekly study time by <strong>1.5 hours</strong>. ${constraintEn}` : 
            `Haftalık çalışma süresini <strong>1.5 saat</strong> artırın. ${constraintTr}`;
            
        let act2 = currentLang === 'EN' ?
            `Reduce unexcused absences by <strong>${Math.max(1, Math.round(abs/2))} days</strong>.` :
            `Geçersiz devamsızlığı <strong>${Math.max(1, Math.round(abs/2))} gün</strong> azaltın.`;

        let acts = [
            { icon: "fa-book-open", text: act1 },
            { icon: "fa-calendar-check", text: act2 }
        ];

        acts.forEach(act => {
            const li = document.createElement("li");
            li.className = "action-item";
            li.innerHTML = `<i class="fa-solid ${act.icon}"></i> <span>${act.text}</span>`;
            actionList.appendChild(li);
        });

        // Set Fair Math Data
        document.getElementById("new-pass-prob").innerText = "83%";
        
        let costVal = isLowSES ? "8.08" : (isMidSES ? "10.65" : "13.23");
        document.getElementById("recourse-cost").innerText = costVal;
        
        document.getElementById("recourse-rfd").innerText = "0.85";
    }

    resetBtn.addEventListener("click", () => {
        resultPanel.classList.add("hidden");
        studentSelect.value = "";
        isCustom = false;
        document.querySelector(".progress-bar").style.width = '0%';
    });

    // Initialize Default
    applyLanguage('EN');
});
