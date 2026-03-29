package turboquant

import (
	"math"
	"testing"
)

func TestCompressionRatio(t *testing.T) {
	tests := []struct {
		bits    int
		minRatio float64
		maxRatio float64
	}{
		{2, 5.5, 7.0},  // ~6.4x
		{3, 4.5, 5.8},  // ~5.3x
		{4, 3.5, 4.5},  // ~4.0x
	}

	for _, tt := range tests {
		ratio := CompressionRatio(tt.bits)
		if ratio < tt.minRatio || ratio > tt.maxRatio {
			t.Errorf("CompressionRatio(%d) = %.2f, want between %.1f and %.1f",
				tt.bits, ratio, tt.minRatio, tt.maxRatio)
		}
	}
}

func TestParseConfig(t *testing.T) {
	tests := []struct {
		input    string
		enabled  bool
		numBits  int
	}{
		{"", false, DefaultBits},
		{"false", false, DefaultBits},
		{"0", false, DefaultBits},
		{"true", true, 3},
		{"1", true, 3},
		{"2", true, 2},
		{"3", true, 3},
		{"4", true, 4},
		{"unknown", true, 3}, // defaults to 3-bit
	}

	for _, tt := range tests {
		cfg := ParseConfig(tt.input)
		if cfg.Enabled != tt.enabled {
			t.Errorf("ParseConfig(%q).Enabled = %v, want %v", tt.input, cfg.Enabled, tt.enabled)
		}
		if cfg.NumBits != tt.numBits {
			t.Errorf("ParseConfig(%q).NumBits = %d, want %d", tt.input, cfg.NumBits, tt.numBits)
		}
	}
}

func TestConfigValidate(t *testing.T) {
	valid := Config{Enabled: true, NumBits: 3}
	if err := valid.Validate(); err != nil {
		t.Errorf("Validate() returned error for valid config: %v", err)
	}

	invalid := Config{Enabled: true, NumBits: 8}
	if err := invalid.Validate(); err == nil {
		t.Error("Validate() should return error for NumBits=8")
	}

	disabled := Config{Enabled: false, NumBits: 99}
	if err := disabled.Validate(); err != nil {
		t.Error("Validate() should not return error when disabled")
	}
}

func TestEffectiveCompressionRatio(t *testing.T) {
	disabled := Config{Enabled: false}
	if r := disabled.EffectiveCompressionRatio(); r != 1.0 {
		t.Errorf("disabled config ratio = %f, want 1.0", r)
	}

	enabled := Config{Enabled: true, NumBits: 3}
	r := enabled.EffectiveCompressionRatio()
	if r < 4.0 || r > 6.0 {
		t.Errorf("3-bit ratio = %f, want between 4.0 and 6.0", r)
	}
}

func TestBytesPerElement(t *testing.T) {
	disabled := Config{Enabled: false}
	if b := disabled.BytesPerElement(); b != 2.0 {
		t.Errorf("disabled bytes/elem = %f, want 2.0", b)
	}

	enabled := Config{Enabled: true, NumBits: 3}
	b := enabled.BytesPerElement()
	if b >= 2.0 || b <= 0.0 {
		t.Errorf("3-bit bytes/elem = %f, want between 0 and 2.0", b)
	}

	// Verify bytes/elem is consistent with compression ratio
	ratio := enabled.EffectiveCompressionRatio()
	expected := 2.0 / ratio
	if math.Abs(b-expected) > 0.001 {
		t.Errorf("bytes/elem %f inconsistent with ratio %f (expected %f)", b, ratio, expected)
	}
}
