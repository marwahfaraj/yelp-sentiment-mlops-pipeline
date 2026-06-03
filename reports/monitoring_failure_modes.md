# Monitoring Failure Modes and Best Practices

## Failure Modes

| Category | Failure Mode | Project Monitor | Mitigation |
|---|---|---|---|
| traditional_software | logic_error | Notebook/script validation, report sanity checks, model quality metrics | Review failed cells/logs and rerun validation before deployment. |
| traditional_software | integration_or_deployment_error | SageMaker training job status and Batch Transform job status | CloudWatch alarm if training or transform job does not complete. |
| traditional_software | dependency_change | requirements.txt plus SageMaker training logs | Pin compatible SageMaker/scikit-learn versions for reproducible jobs. |
| traditional_software | hardware_or_downtime | SageMaker job status and CloudWatch infrastructure metrics | Alert on failed jobs and inspect CloudWatch logs/metrics. |
| ml_specific | data_distribution_shift | Review length drift, class balance drift, term frequency drift | Investigate input changes and retrain if drift persists. |
| ml_specific | edge_cases | Missing/empty text rate and outlier review length checks | Add preprocessing guards and collect examples for retraining. |
| ml_specific | degenerate_feedback_loop | Prediction distribution and business proxy metrics over time | Review whether predictions influence future review sampling or business decisions. |

## Monitoring Best Practices

- Start with a small number of critical metrics: macro F1, missing text rate, class drift, and job success.
- Use CloudWatch dashboards and alarms so failures are visible without reading notebook outputs.
- Include context in reports: model artifact path, batch output path, thresholds, and status.
- Review thresholds regularly to reduce alert fatigue.
- Protect monitoring artifacts because review text and identifiers may contain sensitive information.
- Collaborate with the team to decide which business metric should be monitored in the final version.
