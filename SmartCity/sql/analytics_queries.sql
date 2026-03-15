SELECT *
FROM environment_data
ORDER BY pipeline_run_at DESC
LIMIT 10;

SELECT
    DATE(pipeline_run_at) AS day,
    AVG(temperature) AS avg_temperature,
    AVG(aqi) AS avg_aqi,
    AVG(pm2_5) AS avg_pm2_5,
    AVG(pm10) AS avg_pm10
FROM environment_data
GROUP BY DATE(pipeline_run_at)
ORDER BY day;

SELECT
    hour,
    AVG(aqi) AS avg_aqi,
    AVG(pm2_5) AS avg_pm2_5,
    AVG(pm10) AS avg_pm10,
    COUNT(*) AS observations
FROM environment_data
GROUP BY hour
ORDER BY hour;

SELECT
    weather_main,
    AVG(aqi) AS avg_aqi,
    AVG(pm2_5) AS avg_pm2_5,
    AVG(pm10) AS avg_pm10,
    AVG(temperature) AS avg_temperature,
    COUNT(*) AS observations
FROM environment_data
GROUP BY weather_main
ORDER BY avg_aqi DESC;

SELECT
    DATE(pipeline_run_at) AS day,
    MAX(aqi) AS max_aqi,
    MAX(pm2_5) AS max_pm2_5,
    MAX(pm10) AS max_pm10
FROM environment_data
GROUP BY DATE(pipeline_run_at)
ORDER BY max_aqi DESC, max_pm2_5 DESC;

SELECT
    city,
    temperature,
    humidity,
    wind_speed,
    aqi,
    pm2_5,
    pm10,
    pollution_load,
    pipeline_run_at
FROM environment_data
ORDER BY pollution_load DESC
LIMIT 10;