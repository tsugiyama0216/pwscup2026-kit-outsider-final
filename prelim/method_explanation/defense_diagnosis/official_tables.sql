-- Original official workbook rows, loaded unchanged into an in-memory SQLite table.
-- MIA contributions: denominator is all 20 attackers of target24, including zero.
SELECT 'チーム' || attacker AS attacker, attacker AS attacker_id,
       t_j AS tpr, tpr_all AS all_tpr, tpr_rare AS rare_tpr,
       members, rare_members,
       t_j / (SELECT COUNT(*) FROM official_mia WHERE target = 24) AS raw_average_contribution
FROM official_mia
WHERE target = 24 AND t_j > 0
ORDER BY t_j DESC, attacker;

-- Rare denominators and official interpolated TPR. No individual records displayed.
SELECT attacker, members, rare_members, tpr_all AS all_tpr, tpr_rare AS rare_tpr
FROM official_mia
WHERE target = 24 AND rare_members > 0
ORDER BY tpr_rare DESC, attacker;

-- Official target/control rates plus an independently counted constant-10 baseline.
WITH comparison AS (
    SELECT 'チーム' || attacker AS method,
           CAST(ROUND(p_m * 163) AS INTEGER) AS m_correct,
           CAST(ROUND(p_c * 163) AS INTEGER) AS c_correct,
           p_m AS m_rate, p_c AS c_rate, loss
    FROM official_aia WHERE target = 24 AND attacker IN (5, 19)
    UNION ALL
    SELECT '全員に10年',
           SUM(CASE WHEN label = 'member' AND ABS(time - 10) <= 0.5 THEN 1 ELSE 0 END),
           SUM(CASE WHEN label = 'control' AND ABS(time - 10) <= 0.5 THEN 1 ELSE 0 END),
           AVG(CASE WHEN label = 'member' THEN CASE WHEN ABS(time - 10) <= 0.5 THEN 1.0 ELSE 0.0 END END),
           AVG(CASE WHEN label = 'control' THEN CASE WHEN ABS(time - 10) <= 0.5 THEN 1.0 ELSE 0.0 END END),
           NULL
    FROM aia_truth_24
)
SELECT method, m_correct, c_correct, m_rate, c_rate,
       COALESCE(loss, MIN(1.0, MAX(0.0, (m_rate - c_rate) / (1 - c_rate)))) AS loss
FROM comparison ORDER BY loss DESC;
