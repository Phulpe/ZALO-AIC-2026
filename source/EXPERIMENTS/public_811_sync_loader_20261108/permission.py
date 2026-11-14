"""Explicit evidence policy; user attestation is never labeled independent verification."""
def permitted(evidence):
    if not isinstance(evidence,dict):return False
    if evidence.get('status')=='verified':return True
    return (evidence.get('status')=='user_attested' and
            evidence.get('accepted_for_this_run') is True and
            bool(evidence.get('user_statement')) and bool(evidence.get('agents_sha256')))
