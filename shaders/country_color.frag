#version 150

// Inputs from vertex shader
in vec2 texcoord;
in vec3 worldNormal;
in vec3 worldPos;
in vec3 objectPos;

// The base texture on the model
uniform sampler2D p3d_Texture0;

// Custom uniforms for country coloring
uniform vec4 countryColor;      // The tint color (r, g, b, blend_strength)
uniform float colorBlendMode;   // 0 = multiply, 1 = overlay/blend, 2 = additive tint
uniform float animState;        // 0 = idle, 1 = selected, 2 = neighbor/reachable, 3 = hover
uniform float countryTime;      // Global time for animations

// Output
out vec4 fragColor;

// --- Animated effects ---

// Gentle pulsing glow for the selected country
vec3 selectedPulse(vec3 base, vec3 tint, float t) {
    // Slow sine pulse between 0.6 and 1.0 brightness
    float pulse = 0.8 + 0.2 * sin(t * 2.5);
    return mix(base, tint * pulse, 0.7);
}

// Ripple/wave effect radiating outward for neighbor countries (beckoning)
vec3 neighborWave(vec3 base, vec3 tint, float t, vec2 uv) {
    // Radial wave from center of the mesh UV (0.5, 0.5)
    float dist = length(uv - vec2(0.5));
    float wave = 0.5 + 0.5 * sin(dist * 20.0 - t * 3.0);
    // Blend the wave pattern with the tint
    float strength = mix(0.3, 0.7, wave);
    return mix(base, tint, strength);
}

// Bright shimmer for hover
vec3 hoverShimmer(vec3 base, vec3 tint, float t, vec2 uv) {
    // Diagonal sweep highlight
    float sweep = sin((uv.x + uv.y) * 8.0 - t * 4.0) * 0.5 + 0.5;
    float highlight = smoothstep(0.4, 0.6, sweep);
    vec3 bright = tint * (1.0 + highlight * 0.5);
    return mix(base, bright, 0.55);
}

void main() {
    // Sample the base texture
    vec4 baseColor = texture(p3d_Texture0, texcoord) * 0.25;
    baseColor.rgb = baseColor.rrr; // Convert to grayscale

    // Extract blend strength from the alpha of countryColor
    float blendStrength = countryColor.a;
    vec3 tint = countryColor.rgb;

    vec3 result;

    if (colorBlendMode < 0.5) {
        // Mode 0: Multiply blend
        vec3 multiplied = baseColor.rgb * tint;
        result = mix(baseColor.rgb, multiplied, blendStrength);
    } else if (colorBlendMode < 1.5) {
        // Mode 1: Overlay blend
        float luma = dot(baseColor.rgb, vec3(0.299, 0.587, 0.114));
        vec3 tinted = tint * luma;
        result = mix(baseColor.rgb, tinted, blendStrength);
    } else {
        // Mode 2: Additive tint
        vec3 added = baseColor.rgb + tint * blendStrength;
        result = clamp(added, 0.0, 1.0);
    }

    // Apply animation based on animState
    if (animState > 0.5 && animState < 1.5) {
        // State 1: Selected — pulsing glow
        result = selectedPulse(result, tint, countryTime);
    } else if (animState > 1.5 && animState < 2.5) {
        // State 2: Neighbor/reachable — ripple wave beckoning the player
        result = neighborWave(result, tint, countryTime, texcoord);
    } else if (animState > 2.5 && animState < 3.5) {
        // State 3: Hover — shimmer highlight
        result = hoverShimmer(result, tint, countryTime, texcoord);
    }
    // State 0: idle — no animation, just the static tint

    fragColor = vec4(result, baseColor.a + 0.2);
}
