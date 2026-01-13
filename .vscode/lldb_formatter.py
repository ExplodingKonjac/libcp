import lldb

def ModintSummary(valobj, internal_dict):
    val_member = valobj.GetChildMemberWithName("_val")
    if not val_member.IsValid():
        val_member = valobj.GetChildAtIndex(0).GetChildMemberWithName("_val")

    if not val_member.IsValid():
        return "<Error: no _val>"

    raw_val = val_member.GetValueAsUnsigned(0)
    options = lldb.SBExpressionOptions()
    options.SetFetchDynamicValue(lldb.eDynamicCanRunTarget)
    options.SetTimeoutInMicroSeconds(100000)

    res = valobj.EvaluateExpression("(*this)()", options)
    if res.GetError().Success():
        return res.GetValue()
    return f"raw={raw_val} (Err)"

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('type summary add -x ".*SModint<.+>" -F lldb_formatter.ModintSummary -w cp_modint')
    debugger.HandleCommand('type summary add -x ".*DModint" -F lldb_formatter.ModintSummary -w cp_modint')

    debugger.HandleCommand('type category enable cp_modint')
