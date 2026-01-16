Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\RAG-App\RAG-System-Portable"
WshShell.Run chr(34) & "C:\RAG-App\RAG-System-Portable\RAG System.bat" & chr(34), 0, False
Set WshShell = Nothing
