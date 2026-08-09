# Keep Windows awake for the duration of an unattended campaign — the caffeinate / systemd-inhibit
# equivalent that driver.sh looks for on macOS and Linux.
#
# Uses SetThreadExecutionState, which is process-scoped: the request lives exactly as long as this
# process and Windows drops it the moment the process exits. Nothing global is reconfigured, so a
# crashed or killed driver cannot leave the machine unable to sleep.
#
# ES_CONTINUOUS (0x80000000) makes the state stick until reset; ES_SYSTEM_REQUIRED (0x00000001)
# blocks idle sleep. The display is deliberately NOT held on (no ES_DISPLAY_REQUIRED) — the screen
# may blank, the machine just must not suspend. Lid-close sleep is a hardware policy and is not
# overridable this way, same caveat as caffeinate on a MacBook.
#
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File autoresearch/keepawake.ps1
#         (runs until killed)

$signature = @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$power = Add-Type -MemberDefinition $signature -Name PowerUtil -Namespace Campaign -PassThru

$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'

$prev = $power::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
if ($prev -eq 0) {
    Write-Error "SetThreadExecutionState failed; the machine may sleep mid-campaign."
    exit 1
}
Write-Output "keepawake: idle sleep blocked (pid $PID). Kill this process to release."

try {
    while ($true) { Start-Sleep -Seconds 60 }
}
finally {
    # Release explicitly on a clean exit; Windows would do it anyway when the process dies.
    [void]$power::SetThreadExecutionState($ES_CONTINUOUS)
}
