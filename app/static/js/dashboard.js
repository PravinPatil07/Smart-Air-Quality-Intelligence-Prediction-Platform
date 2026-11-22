
"use strict";

// ============================================================
// AIRSENSE INDIA - FRONTEND CONTROLLER
// ============================================================

const API = "/api";

let indiaMap = null;
let mapMarkers = [];

let dashboardInitialized = false;


// ============================================================
// DOM HELPERS
// ============================================================

function getElement(id) {

    return document.getElementById(id);
}


function showLoading(show) {

    getElement("loadingOverlay")
        ?.classList.toggle("hidden", !show);
}


function showDashboardError(message) {

    const element = getElement("dashboardError");

    if (!element) return;

    element.textContent = message;

    element.classList.remove("hidden");
}


function hideDashboardError() {

    getElement("dashboardError")
        ?.classList.add("hidden");
}


function showPredictionError(message) {

    const element = getElement("predictionError");

    if (!element) return;

    element.textContent = message;

    element.classList.remove("hidden");
}


function hidePredictionError() {

    getElement("predictionError")
        ?.classList.add("hidden");
}


function formatNumber(value, decimals = 1) {

    if (
        value === null ||
        value === undefined ||
        value === "" ||
        !Number.isFinite(Number(value))
    ) {
        return "--";
    }

    return Number(value).toLocaleString("en-IN", {
        minimumFractionDigits: 0,
        maximumFractionDigits: decimals
    });
}


function formatR2(value) {

    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(Number(value))
    ) {
        return "--";
    }

    let number = Number(value);

    // Support both decimal R² and percentage formats.
    if (Math.abs(number) <= 1) {
        number *= 100;
    }

    return `${number.toFixed(2)}%`;
}


function escapeHTML(value) {

    return String(value).replace(/[&<>"']/g, character => {

        const replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        };

        return replacements[character];

    });
}


// ============================================================
// API HELPER
// ============================================================

async function fetchJSON(url) {

    const response = await fetch(url, {
        method: "GET",
        headers: {
            "Accept": "application/json"
        }
    });

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(
            `Server returned an invalid response (${response.status}).`
        );
    }

    if (!response.ok) {

        throw new Error(
            data.error ||
            `Request failed with status ${response.status}.`
        );
    }

    return data;
}


// ============================================================
// 1. DASHBOARD OVERVIEW
// ============================================================

async function loadOverview() {

    const data = await fetchJSON(
        `${API}/overview`
    );

    getElement("cityCount").textContent =
        formatNumber(data.city_count, 0);

    getElement("recordCount").textContent =
        formatNumber(data.record_count, 0);

    getElement("averageAqi").textContent =
        formatNumber(data.average_aqi, 1);

    getElement("latestDate").textContent =
        data.latest_date || "--";

    renderIndiaMap(data.cities || []);

}


// ============================================================
// 2. CITY DROPDOWNS
// ============================================================

async function loadCities() {

    const data = await fetchJSON(
        `${API}/cities`
    );

    const allCities = data.cities || [];

    const predictionCities =
        data.prediction_cities || [];

    const predictionSelect =
        getElement("citySelect");

    const analyticsSelect =
        getElement("analyticsCity");


    // ---------------------------------------------
    // Prediction dropdown
    // Only trained cities are allowed.
    // ---------------------------------------------

    predictionSelect.innerHTML = "";

    if (predictionCities.length === 0) {

        predictionSelect.innerHTML = `
            <option value="">
                No supported prediction cities
            </option>
        `;

        predictionSelect.disabled = true;

        getElement("predictButton").disabled = true;

    } else {

        predictionCities.forEach(city => {

            const option = document.createElement("option");

            option.value = city;
            option.textContent = city;

            predictionSelect.appendChild(option);

        });

        predictionSelect.disabled = false;

        getElement("predictButton").disabled = false;

    }


    // ---------------------------------------------
    // Historical analytics dropdown
    // All dataset cities are allowed.
    // ---------------------------------------------

    analyticsSelect.innerHTML = "";

    if (allCities.length === 0) {

        analyticsSelect.innerHTML = `
            <option value="">
                No cities available
            </option>
        `;

        analyticsSelect.disabled = true;

    } else {

        allCities.forEach(city => {

            const option = document.createElement("option");

            option.value = city;
            option.textContent = city;

            analyticsSelect.appendChild(option);

        });

        analyticsSelect.disabled = false;

        await loadCityAnalytics(allCities[0]);

    }

}


// ============================================================
// 3. INDIA MAP
// ============================================================

function renderIndiaMap(cities) {

    const mapElement = getElement("indiaMap");

    if (!mapElement) return;

    if (typeof L === "undefined") {

        mapElement.innerHTML =
            "Map library could not be loaded. Check your internet connection.";

        return;
    }


    // Remove old map instance if one exists.
    if (indiaMap !== null) {

        indiaMap.remove();

        indiaMap = null;

        mapMarkers = [];

    }


    indiaMap = L.map("indiaMap", {
        scrollWheelZoom: false
    }).setView([22.5, 79], 4.5);


    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution: "&copy; OpenStreetMap contributors",
            maxZoom: 18
        }
    ).addTo(indiaMap);


    cities.forEach(city => {

        if (
            !Array.isArray(city.coordinates) ||
            city.coordinates.length !== 2
        ) {
            return;
        }


        const marker = L.circleMarker(
            city.coordinates,
            {
                radius: 8,

                fillColor: city.color || "#38d9c0",

                color: "#ffffff",

                weight: 1,

                opacity: 1,

                fillOpacity: 0.85
            }
        ).addTo(indiaMap);


        const popup = `
            <div class="map-popup">

                <strong>${escapeHTML(city.city)}</strong>

                <br>

                AQI:
                ${formatNumber(city.aqi, 1)}

                <br>

                Category:
                ${escapeHTML(city.category)}

                <br>

                <small>
                    Observation:
                    ${escapeHTML(city.date)}
                </small>

            </div>
        `;


        marker.bindPopup(popup);


        marker.on("click", () => {

            const analyticsSelect =
                getElement("analyticsCity");

            const predictionSelect =
                getElement("citySelect");


            // City is available in historical analytics.
            analyticsSelect.value = city.city;

            loadCityAnalytics(city.city);


            // Only select it for prediction if the city
            // exists in the trained-city dropdown.
            const predictionOption = Array.from(
                predictionSelect.options
            ).find(option => option.value === city.city);

            if (predictionOption) {

                predictionSelect.value = city.city;

            }

        });


        mapMarkers.push(marker);

    });


    // Fix map rendering when its container is initialized.
    setTimeout(() => {

        if (indiaMap) {
            indiaMap.invalidateSize();
        }

    }, 150);

}


// ============================================================
// 4. NEXT-DAY AQI PREDICTION
// ============================================================

async function predictAQI() {

    const city = getElement("citySelect").value;

    const button = getElement("predictButton");

    const resultBox = getElement("predictionResult");


    if (!city) {

        showPredictionError(
            "Please select a supported prediction city."
        );

        return;
    }


    button.disabled = true;

    button.innerHTML = `
        <span class="spinner-border spinner-border-sm"></span>
        Predicting...
    `;


    hidePredictionError();

    resultBox.classList.add("hidden");


    try {

        const data = await fetchJSON(
            `${API}/predict/${encodeURIComponent(city)}`
        );


        // -----------------------------------------
        // Main prediction
        // -----------------------------------------

        getElement("predictedAqi").textContent =
            formatNumber(data.predicted_aqi, 1);


        getElement("predictedCategory").textContent =
            data.category || "Unavailable";


        getElement("predictedCategory").style.color =
            data.color || "#38d9c0";


        // -----------------------------------------
        // Supporting details
        // -----------------------------------------

        getElement("predictionCity").textContent =
            data.city || "--";


        getElement("observationDate").textContent =
            data.observation_date || "--";


        getElement("forecastDate").textContent =
            data.forecast_date || "--";


        getElement("latestObservedAqi").textContent =
            formatNumber(data.latest_observed_aqi, 1);


        getElement("lowerBound").textContent =
            data.lower_90 === null
                ? "Unavailable"
                : formatNumber(data.lower_90, 1);


        getElement("upperBound").textContent =
            data.upper_90 === null
                ? "Unavailable"
                : formatNumber(data.upper_90, 1);


        getElement("predictionNote").textContent =
            data.interval_note || "";


        resultBox.classList.remove("hidden");


        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });


    } catch (error) {

        showPredictionError(error.message);

    } finally {

        button.disabled = false;

        button.innerHTML = `
            <i class="bi bi-lightning-charge"></i>
            Predict AQI
        `;

    }

}


// ============================================================
// 5. CITY HISTORICAL ANALYTICS
// ============================================================

async function loadCityAnalytics(city) {

    if (!city) return;

    try {

        const data = await fetchJSON(
            `${API}/city/${encodeURIComponent(city)}`
        );

        renderAQIChart(data);

        renderPollutantChart(data);


    } catch (error) {

        console.error(
            "Historical analytics error:",
            error
        );

        renderEmptyChart(
            "aqiChart",
            "Unable to load AQI history."
        );

        renderEmptyChart(
            "pollutantChart",
            "Unable to load pollutant information."
        );

    }

}


// ============================================================
// 6. AQI HISTORICAL TREND CHART
// ============================================================

function renderAQIChart(data) {

    const history = data.history || [];

    if (history.length === 0) {

        renderEmptyChart(
            "aqiChart",
            "No valid AQI observations available for this city."
        );

        return;
    }


    const dates = history.map(item => item.date);

    const values = history.map(item => item.aqi);


    const trace = {

        x: dates,

        y: values,

        type: "scatter",

        mode: "lines",

        name: "AQI",

        line: {
            color: "#38d9c0",
            width: 2.5,
            shape: "linear"
        },

        fill: "tozeroy",

        fillcolor: "rgba(56,217,192,0.07)"

    };


    const layout = {

        paper_bgcolor: "transparent",

        plot_bgcolor: "transparent",

        margin: {
            l: 50,
            r: 15,
            t: 20,
            b: 55
        },

        xaxis: {

            color: "#8796aa",

            gridcolor: "rgba(148,163,184,0.10)",

            title: "Date"

        },

        yaxis: {

            color: "#8796aa",

            gridcolor: "rgba(148,163,184,0.10)",

            title: "AQI"

        },

        font: {
            color: "#e8eef7",
            family: "DM Sans"
        },

        showlegend: false,

        hovermode: "x unified"

    };


    Plotly.react(
        "aqiChart",
        [trace],
        layout,
        {
            responsive: true,
            displayModeBar: false
        }
    );

}


// ============================================================
// 7. POLLUTANT PROFILE CHART
// ============================================================

function renderPollutantChart(data) {

    const pollutants = data.pollutants || {};

    const names = Object.keys(pollutants);

    const values = Object.values(pollutants);


    if (names.length === 0) {

        renderEmptyChart(
            "pollutantChart",
            "No pollutant measurements available for this city."
        );

        return;
    }


    const trace = {

        x: names,

        y: values,

        type: "bar",

        marker: {
            color: [
                "#38d9c0",
                "#5b8cff",
                "#a78bfa",
                "#f59e0b",
                "#fb7185",
                "#22d3ee",
                "#84cc16"
            ]
        },

        hovertemplate:
            "%{x}<br>Mean value: %{y:.3f}<extra></extra>"

    };


    const layout = {

        paper_bgcolor: "transparent",

        plot_bgcolor: "transparent",

        margin: {
            l: 50,
            r: 15,
            t: 20,
            b: 50
        },

        xaxis: {

            color: "#8796aa",

            gridcolor: "rgba(148,163,184,0.10)"

        },

        yaxis: {

            color: "#8796aa",

            gridcolor: "rgba(148,163,184,0.10)",

            title: "Mean recorded value"

        },

        font: {
            color: "#e8eef7",
            family: "DM Sans"
        },

        showlegend: false

    };


    Plotly.react(
        "pollutantChart",
        [trace],
        layout,
        {
            responsive: true,
            displayModeBar: false
        }
    );

}


// ============================================================
// 8. EMPTY CHART HELPER
// ============================================================

function renderEmptyChart(elementId, message) {

    if (typeof Plotly === "undefined") {
        return;
    }

    const layout = {

        paper_bgcolor: "transparent",

        plot_bgcolor: "transparent",

        margin: {
            l: 20,
            r: 20,
            t: 20,
            b: 20
        },

        xaxis: {
            visible: false
        },

        yaxis: {
            visible: false
        },

        annotations: [
            {
                text: escapeHTML(message),

                x: 0.5,

                y: 0.5,

                xref: "paper",

                yref: "paper",

                showarrow: false,

                font: {
                    color: "#8796aa",
                    size: 12
                }
            }
        ]

    };


    Plotly.react(
        elementId,
        [],
        layout,
        {
            responsive: true,
            displayModeBar: false
        }
    );

}


// ============================================================
// 9. MODEL PERFORMANCE
// ============================================================

async function loadModelInformation() {

    const data = await fetchJSON(
        `${API}/model`
    );


    const metrics = data.test_metrics || {};


    getElement("modelMae").textContent =
        formatNumber(metrics.MAE, 2);


    getElement("modelRmse").textContent =
        formatNumber(metrics.RMSE, 2);


    getElement("modelR2").textContent =
        formatR2(metrics.R2);


    getElement("featureCount").textContent =
        formatNumber(data.feature_count, 0);


    renderFeatureChart(
        data.top_features || []
    );

}


// ============================================================
// 10. PERMUTATION FEATURE IMPORTANCE
// ============================================================

function renderFeatureChart(features) {

    const message = getElement("featureChartMessage");

    if (!features || features.length === 0) {

        renderEmptyChart(
            "featureChart",
            "Feature importance data is unavailable."
        );

        message.textContent =
            "Check reports/statistics/phase4_2_permutation_importance.csv.";

        return;
    }


    // The API returns features in descending importance order.
    // Reverse them so the largest bar appears at the top.

    const topFeatures = [...features]
        .slice(0, 10)
        .reverse();


    const featureNames = topFeatures.map(
        item => item.feature
    );

    const importanceValues = topFeatures.map(
        item => item.importance
    );


    const trace = {

        x: importanceValues,

        y: featureNames,

        type: "bar",

        orientation: "h",

        marker: {
            color: "#38d9c0"
        },

        hovertemplate:
            "%{y}<br>Importance: %{x:.4f}<extra></extra>"

    };


    const layout = {

        paper_bgcolor: "transparent",

        plot_bgcolor: "transparent",

        margin: {
            l: 150,
            r: 20,
            t: 15,
            b: 50
        },

        xaxis: {

            title: "Increase in validation RMSE",

            color: "#8796aa",

            gridcolor: "rgba(148,163,184,0.10)"

        },

        yaxis: {

            color: "#8796aa",

            automargin: true

        },

        font: {
            color: "#e8eef7",
            family: "DM Sans"
        },

        showlegend: false

    };


    Plotly.react(
        "featureChart",
        [trace],
        layout,
        {
            responsive: true,
            displayModeBar: false
        }
    );


    message.textContent =
        "Feature importance represents predictive contribution " +
        "to validation performance, not causation.";

}


// ============================================================
// 11. MOBILE NAVIGATION
// ============================================================

function initializeNavigation() {

    const menu = getElement("mobileMenu");

    const sidebar = getElement("sidebar");


    if (!menu || !sidebar) {
        return;
    }


    menu.addEventListener("click", () => {

        sidebar.classList.toggle("open");

    });


    document.querySelectorAll(".sidebar-nav a")
        .forEach(link => {

            link.addEventListener("click", () => {

                sidebar.classList.remove("open");

            });

        });

}


// ============================================================
// 12. EVENT LISTENERS
// ============================================================

function initializeEventListeners() {

    getElement("predictButton")
        ?.addEventListener("click", predictAQI);


    getElement("analyticsCity")
        ?.addEventListener("change", event => {

            loadCityAnalytics(event.target.value);

        });


    getElement("citySelect")
        ?.addEventListener("change", event => {

            const city = event.target.value;

            const analyticsSelect =
                getElement("analyticsCity");


            // Synchronize analytics if the selected prediction
            // city is also available in the historical dataset.

            const matchingOption = Array.from(
                analyticsSelect.options
            ).find(option => option.value === city);


            if (matchingOption) {

                analyticsSelect.value = city;

                loadCityAnalytics(city);

            }

        });


    initializeNavigation();

}


// ============================================================
// 13. APPLICATION INITIALIZATION
// ============================================================

async function initializeDashboard() {

    if (dashboardInitialized) {
        return;
    }

    dashboardInitialized = true;

    showLoading(true);

    hideDashboardError();


    try {

        // Load dashboard data.
        await loadOverview();

        // Load historical and prediction city lists.
        await loadCities();

        // Load model metrics and feature importance.
        await loadModelInformation();


    } catch (error) {

        console.error(
            "[AirSense] Dashboard initialization failed:",
            error
        );

        showDashboardError(
            `Unable to initialize AirSense: ${error.message}`
        );

    } finally {

        showLoading(false);

    }


    initializeEventListeners();

}


// ============================================================
// START APPLICATION
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    initializeDashboard
);
