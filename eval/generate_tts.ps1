Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$finalPath = Join-Path $PSScriptRoot "command_test_full_6step.wav"
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($finalPath, $format)
$synth.Speak("Pick up the plate, place it on the table, then hand the spoon to Arm A and place it beside the plate.")
$synth.Dispose()

Write-Host "Audio generated successfully: $finalPath"
