uniform bool skirmishRangeActive;
uniform vec2 skirmishRangeCenter;
uniform vec3 skirmishRangeLimits;

vec3 skirmishRangeOverlay(vec3 ground, vec2 worldPosition) {
    vec2 offset = worldPosition - skirmishRangeCenter;
    float distance = length(offset);
    float normal = skirmishRangeLimits.x;
    float march = skirmishRangeLimits.y;
    float charge = skirmishRangeLimits.z;
    if (distance > max(march, charge)) return ground;

    vec3 normalColor = vec3(0.3, 1.0, 0.65);
    vec3 marchColor = vec3(1.0, 0.72, 0.25);
    vec3 chargeColor = vec3(0.15, 0.85, 1.0);
    vec3 result = ground;
    if (distance <= march) {
        vec3 tint = distance <= normal ? normalColor : marchColor;
        result = mix(ground, tint, 0.28);
        float edge = min(abs(distance - normal), abs(distance - march));
        result = mix(result, tint, 0.8 * (1.0 - smoothstep(0.035, 0.09, edge)));
    } else if (charge > march) {
        float stripe = step(0.65, fract((worldPosition.x + worldPosition.y) * 1.5));
        result = mix(ground, chargeColor, 0.12 + 0.12 * stripe);
    }
    if (charge > 0.0 && distance <= charge) {
        float dash = step(0.25, fract(atan(offset.y, offset.x) * 12.0));
        float edge = 1.0 - smoothstep(0.04, 0.14, charge - distance);
        result = mix(result, chargeColor, edge * dash);
    }
    return result;
}