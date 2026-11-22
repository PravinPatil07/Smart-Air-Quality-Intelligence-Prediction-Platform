
-- ==========================================
-- AIRSENSE INDIA: PHASE 2 SQL ANALYSIS
-- ==========================================

-- 1. Total records and cities

SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT City) AS total_cities
FROM city_day;


-- 2. Top 10 cities by average AQI
-- Minimum 30 valid observations

SELECT
    City,
    COUNT(AQI) AS observations,
    ROUND(AVG(AQI), 2) AS average_aqi,
    MAX(AQI) AS maximum_aqi
FROM city_day
WHERE AQI IS NOT NULL
GROUP BY City
HAVING COUNT(AQI) >= 30
ORDER BY average_aqi DESC
LIMIT 10;


-- 3. Lowest average AQI cities

SELECT
    City,
    COUNT(AQI) AS observations,
    ROUND(AVG(AQI), 2) AS average_aqi
FROM city_day
WHERE AQI IS NOT NULL
GROUP BY City
HAVING COUNT(AQI) >= 30
ORDER BY average_aqi ASC
LIMIT 10;


-- 4. Monthly AQI trend

SELECT
    strftime('%Y-%m', Date) AS month,
    ROUND(AVG(AQI), 2) AS average_aqi,
    COUNT(AQI) AS observations
FROM city_day
WHERE AQI IS NOT NULL
GROUP BY month
ORDER BY month;


-- 5. Maharashtra city analysis

SELECT
    City,
    COUNT(AQI) AS observations,
    ROUND(AVG(AQI), 2) AS average_aqi,
    MAX(AQI) AS maximum_aqi
FROM city_day
WHERE City IN (
    'Mumbai',
    'Pune',
    'Nashik',
    'Nagpur',
    'Thane',
    'Navi Mumbai',
    'Aurangabad',
    'Chhatrapati Sambhaji Nagar'
)
GROUP BY City
ORDER BY average_aqi DESC;


-- 6. Annual pollution trends

SELECT
    strftime('%Y', Date) AS year,
    ROUND(AVG(AQI), 2) AS average_aqi,
    ROUND(AVG("PM2.5"), 2) AS average_pm25,
    ROUND(AVG("PM10"), 2) AS average_pm10
FROM city_day
GROUP BY year
ORDER BY year;


-- 7. AQI classification distribution

SELECT
    AQI_Bucket,
    COUNT(*) AS records,
    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*)
         FROM city_day
         WHERE AQI_Bucket IS NOT NULL),
        2
    ) AS percentage
FROM city_day
WHERE AQI_Bucket IS NOT NULL
GROUP BY AQI_Bucket
ORDER BY records DESC;


-- 8. Highest recorded pollution days

SELECT
    City,
    Date,
    AQI,
    "PM2.5",
    "PM10",
    NO2,
    CO
FROM city_day
WHERE AQI IS NOT NULL
ORDER BY AQI DESC
LIMIT 20;
