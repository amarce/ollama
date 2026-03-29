package main

import (
	_ "embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

//go:embed ollama.exe
var ollamaExe []byte

//go:embed app.ico
var appIcon []byte

const version = "2.0.3-turboquant"
const appName = "Ollama"

var (
	advapi32 = syscall.NewLazyDLL("advapi32.dll")
	user32   = syscall.NewLazyDLL("user32.dll")

	regOpenKeyExW    = advapi32.NewProc("RegOpenKeyExW")
	regQueryValueExW = advapi32.NewProc("RegQueryValueExW")
	regSetValueExW   = advapi32.NewProc("RegSetValueExW")
	regCreateKeyExW  = advapi32.NewProc("RegCreateKeyExW")
	regCloseKey      = advapi32.NewProc("RegCloseKey")

	sendMessageTimeoutW = user32.NewProc("SendMessageTimeoutW")
)

const (
	HKEY_CURRENT_USER uintptr = 0x80000001
	KEY_READ          uint32  = 0x20019
	KEY_ALL_ACCESS    uint32  = 0xF003F
	REG_SZ            uint32  = 1
	REG_EXPAND_SZ     uint32  = 2
	ERROR_SUCCESS     uintptr = 0
	HWND_BROADCAST    uintptr = 0xFFFF
	WM_SETTINGCHANGE  uint32  = 0x001A
	SMTO_ABORTIFHUNG  uint32  = 0x0002
)

func main() {
	silent := false
	installDir := ""

	for _, arg := range os.Args[1:] {
		a := strings.ToUpper(arg)
		if a == "/VERYSILENT" || a == "/SILENT" || a == "/S" {
			silent = true
		}
		// Support /DIR=path for custom install directory (matches Inno Setup)
		if strings.HasPrefix(strings.ToUpper(arg), "/DIR=") {
			installDir = arg[5:]
		}
	}

	if installDir == "" {
		installDir = filepath.Join(os.Getenv("LOCALAPPDATA"), "Programs", appName)
	}

	if !silent {
		fmt.Println("==============================================")
		fmt.Printf("  %s TurboQuant Installer v%s\n", appName, version)
		fmt.Println("  Google TurboQuant KV Cache Compression")
		fmt.Println("==============================================")
		fmt.Println()
		fmt.Printf("Install directory: %s\n\n", installDir)
	}

	// Step 1: Kill existing ollama processes
	logMsg(silent, "Stopping existing Ollama processes...")
	taskKill("ollama app.exe")
	taskKill("ollama.exe")

	// Step 2: Create install directory
	if err := os.MkdirAll(installDir, 0755); err != nil {
		exitError(silent, "Error creating directory: %v", err)
	}

	// Step 3: Clean old lib directory (matches [InstallDelete])
	libOllamaDir := filepath.Join(installDir, "lib", "ollama")
	_ = os.RemoveAll(libOllamaDir)
	tempDir := os.Getenv("TEMP")
	if tempDir != "" {
		cleanGlob(filepath.Join(tempDir, "ollama*"))
	}

	// Step 4: Write ollama.exe
	exePath := filepath.Join(installDir, "ollama.exe")
	logMsg(silent, "Installing ollama.exe (%d MB)...", len(ollamaExe)/1024/1024)
	if err := os.WriteFile(exePath, ollamaExe, 0755); err != nil {
		exitError(silent, "Error writing file: %v", err)
	}
	logMsg(silent, "  Written to %s", exePath)

	// Step 5: Write app icon (matches official ISS: Source: ".\assets\app.ico"; DestDir: "{app}")
	iconPath := filepath.Join(installDir, "app.ico")
	if err := os.WriteFile(iconPath, appIcon, 0644); err != nil {
		logMsg(silent, "  Warning: could not write icon: %v", err)
	}

	// Step 6: Add to user PATH via registry
	logMsg(silent, "Configuring PATH...")
	if needsAddPath(installDir) {
		if err := addToPath(installDir); err != nil {
			logMsg(silent, "  Warning: could not update PATH: %v", err)
			logMsg(silent, "  Please add %s to your PATH manually", installDir)
		} else {
			logMsg(silent, "  Added %s to user PATH", installDir)
			broadcastSettingChange()
		}
	} else {
		logMsg(silent, "  PATH already contains %s", installDir)
	}

	// Step 7: Register ollama:// URL protocol
	logMsg(silent, "Registering ollama:// URL protocol...")
	if err := registerURLProtocol(installDir); err != nil {
		logMsg(silent, "  Warning: could not register URL protocol: %v", err)
	} else {
		logMsg(silent, "  Registered ollama:// protocol handler")
	}

	// Step 8: Create Start Menu shortcut (with icon)
	logMsg(silent, "Creating Start Menu shortcut...")
	if err := createShortcut(installDir); err != nil {
		logMsg(silent, "  Warning: could not create shortcut: %v", err)
	} else {
		logMsg(silent, "  Created Start Menu shortcut")
	}

	// Step 9: Clean old update markers
	ollamaDataDir := filepath.Join(os.Getenv("LOCALAPPDATA"), "Ollama")
	_ = os.Remove(filepath.Join(ollamaDataDir, "updates"))

	if !silent {
		fmt.Println()
		fmt.Println("Installation complete!")
		fmt.Println()
		fmt.Println("TurboQuant KV cache compression auto-enables on CUDA GPUs.")
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ollama serve          # Start the server")
		fmt.Println("  ollama run gemma3     # Run a model")
	}
}

func logMsg(silent bool, format string, args ...interface{}) {
	if !silent {
		fmt.Printf(format+"\n", args...)
	}
}

func exitError(silent bool, format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func taskKill(name string) {
	cmd := exec.Command("taskkill.exe", "/f", "/im", name)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	_ = cmd.Run()
}

func cleanGlob(pattern string) {
	matches, _ := filepath.Glob(pattern)
	for _, m := range matches {
		_ = os.RemoveAll(m)
	}
}

// --- Registry helpers using Windows API ---

func utf16Ptr(s string) *uint16 {
	p, _ := syscall.UTF16PtrFromString(s)
	return p
}

func needsAddPath(dir string) bool {
	var hKey syscall.Handle
	ret, _, _ := regOpenKeyExW.Call(
		HKEY_CURRENT_USER,
		uintptr(unsafe.Pointer(utf16Ptr(`Environment`))),
		0, uintptr(KEY_READ),
		uintptr(unsafe.Pointer(&hKey)),
	)
	if ret != ERROR_SUCCESS {
		return true
	}
	defer regCloseKey.Call(uintptr(hKey))

	path := regReadString(hKey, "Path")
	if path == "" {
		return true
	}
	return !strings.Contains(
		";"+strings.ToLower(path)+";",
		";"+strings.ToLower(dir)+";",
	)
}

func addToPath(dir string) error {
	var hKey syscall.Handle
	ret, _, _ := regOpenKeyExW.Call(
		HKEY_CURRENT_USER,
		uintptr(unsafe.Pointer(utf16Ptr(`Environment`))),
		0, uintptr(KEY_ALL_ACCESS),
		uintptr(unsafe.Pointer(&hKey)),
	)
	if ret != ERROR_SUCCESS {
		return fmt.Errorf("RegOpenKeyEx failed: %d", ret)
	}
	defer regCloseKey.Call(uintptr(hKey))

	current := regReadString(hKey, "Path")
	newPath := current
	if newPath != "" && !strings.HasSuffix(newPath, ";") {
		newPath += ";"
	}
	newPath += dir

	return regWriteString(hKey, "Path", newPath, REG_EXPAND_SZ)
}

func regReadString(hKey syscall.Handle, name string) string {
	vn := utf16Ptr(name)
	var dataType, bufSize uint32
	regQueryValueExW.Call(uintptr(hKey), uintptr(unsafe.Pointer(vn)),
		0, uintptr(unsafe.Pointer(&dataType)), 0, uintptr(unsafe.Pointer(&bufSize)))
	if bufSize == 0 {
		return ""
	}
	buf := make([]uint16, bufSize/2)
	ret, _, _ := regQueryValueExW.Call(uintptr(hKey), uintptr(unsafe.Pointer(vn)),
		0, uintptr(unsafe.Pointer(&dataType)),
		uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&bufSize)))
	if ret != ERROR_SUCCESS {
		return ""
	}
	return syscall.UTF16ToString(buf)
}

func regWriteString(hKey syscall.Handle, name, value string, regType uint32) error {
	vn := utf16Ptr(name)
	data, _ := syscall.UTF16FromString(value)
	ret, _, _ := regSetValueExW.Call(uintptr(hKey), uintptr(unsafe.Pointer(vn)),
		0, uintptr(regType),
		uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)*2))
	if ret != ERROR_SUCCESS {
		return fmt.Errorf("RegSetValueEx failed: %d", ret)
	}
	return nil
}

func broadcastSettingChange() {
	env := utf16Ptr("Environment")
	sendMessageTimeoutW.Call(
		HWND_BROADCAST, uintptr(WM_SETTINGCHANGE),
		0, uintptr(unsafe.Pointer(env)),
		uintptr(SMTO_ABORTIFHUNG), uintptr(5000), 0,
	)
}

func registerURLProtocol(installDir string) error {
	appExe := filepath.Join(installDir, "ollama app.exe")
	if err := regCreateAndSet(HKEY_CURRENT_USER, `Software\Classes\ollama`, "", "URL:Ollama Protocol"); err != nil {
		return err
	}
	if err := regCreateAndSet(HKEY_CURRENT_USER, `Software\Classes\ollama`, "URL Protocol", ""); err != nil {
		return err
	}
	return regCreateAndSet(HKEY_CURRENT_USER, `Software\Classes\ollama\shell\open\command`, "",
		fmt.Sprintf(`"%s" "%%1"`, appExe))
}

func regCreateAndSet(root uintptr, subKeyPath, valueName, value string) error {
	var hKey syscall.Handle
	var disposition uint32
	ret, _, _ := regCreateKeyExW.Call(
		root, uintptr(unsafe.Pointer(utf16Ptr(subKeyPath))),
		0, 0, 0, uintptr(KEY_ALL_ACCESS), 0,
		uintptr(unsafe.Pointer(&hKey)),
		uintptr(unsafe.Pointer(&disposition)),
	)
	if ret != ERROR_SUCCESS {
		return fmt.Errorf("RegCreateKeyEx failed for %s: %d", subKeyPath, ret)
	}
	defer regCloseKey.Call(uintptr(hKey))

	return regWriteString(hKey, valueName, value, REG_SZ)
}

func createShortcut(installDir string) error {
	startMenu := filepath.Join(os.Getenv("APPDATA"),
		"Microsoft", "Windows", "Start Menu", "Programs")
	lnkPath := filepath.Join(startMenu, "Ollama.lnk")
	target := filepath.Join(installDir, "ollama.exe")
	iconLoc := filepath.Join(installDir, "app.ico")

	ps := fmt.Sprintf(
		`$ws = New-Object -ComObject WScript.Shell; `+
			`$s = $ws.CreateShortcut('%s'); `+
			`$s.TargetPath = '%s'; `+
			`$s.IconLocation = '%s'; `+
			`$s.Description = 'Ollama TurboQuant'; `+
			`$s.Save()`,
		lnkPath, target, iconLoc,
	)
	cmd := exec.Command("powershell.exe", "-NoProfile", "-Command", ps)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	return cmd.Run()
}
