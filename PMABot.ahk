#Persistent
SetTitleMatchMode, 2

; Start recording
Send, {F9}  ; Press F9 to start recording
Sleep, 1000  ; Small delay

Loop
{
    Sleep, 30000  ; Wait 30 seconds (adjust this to change recording length)
    Send, {F10}  ; Stop recording
    Sleep, 1000  ; Small delay
    Send, {F9}  ; Start new recording
}

; Press F12 to manually stop the script
F12::
    Send, {F10}  ; Stop recording
    ExitApp
