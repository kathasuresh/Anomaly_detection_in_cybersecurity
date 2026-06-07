def create_features(
    login_attempts,
    failed_logins
):

    failure_ratio = (
        failed_logins /
        (login_attempts + 1)
    )

    login_risk_score = (
        login_attempts *
        failed_logins
    )

    return (
        failure_ratio,
        login_risk_score
    )