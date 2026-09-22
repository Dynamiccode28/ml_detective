_SUSPICIOUS_NAME_FRAGMENTS=[
    "_after","_post","outcome","result","final","completed","resolution","resolved","closed",
    "end_date","exit_","termination",
]
def has_suspicious_name(column_name:str)->list[str]:
    lowercase_name=column_name.lower()
    return[
        fragment for fragment in _SUSPICIOUS_NAME_FRAGMENTS
        if fragment in lowercase_name
    ]