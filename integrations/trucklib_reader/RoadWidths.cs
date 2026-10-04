using System.Globalization;
using TruckLib;
using TruckLib.Sii;

sealed record RoadBoundary(double left_m, double right_m,
    string source = "sii-road-size", bool validated = false);

sealed class RoadWidths
{
    private readonly Dictionary<string, RoadBoundary> widths = new();
    public string Status { get; private set; } = "No explicit road size definitions";

    public RoadWidths(IFileSystem files)
    {
        int failed = 0;
        if (!files.DirectoryExists("/def/world")) return;
        foreach (var path in files.GetFiles("/def/world")
            .Where(p => Path.GetFileName(p).StartsWith("road_look", StringComparison.Ordinal)
                && p.EndsWith(".sii", StringComparison.Ordinal)).Order())
        {
            try
            {
                foreach (var unit in SiiFile.Open(path, files).Units.Where(u => u.Class == "road_look"))
                {
                    var key = unit.Name.StartsWith("road.") ? unit.Name[5..] : unit.Name;
                    widths.Remove(key);
                    var attrs = unit.Attributes;
                    if (attrs.ContainsKey("template_left") || attrs.ContainsKey("template_right")
                        || attrs.ContainsKey("lane_offsets_left") || attrs.ContainsKey("lane_offsets_right")) continue;
                    if (!Number(attrs, "road_size_left", out var left)
                        || !Number(attrs, "road_size_right", out var right)) continue;
                    if (attrs.ContainsKey("road_offset")
                        && (!Number(attrs, "road_offset", out var offset) || Math.Abs(offset) > 0.001)) continue;
                    if (left < 0 || right < 0 || left > 40 || right > 40 || left+right < 2) continue;
                    widths[key] = new(left, right);
                }
            }
            catch (Exception) { failed++; }
        }
        Status = $"Explicit size definitions: {widths.Count}; unreadable definition files: {failed}; template sizes unknown";
    }

    public RoadBoundary Get(string key) => widths.GetValueOrDefault(key);

    private static bool Number(Dictionary<string, object> attrs, string key, out double value)
    {
        value = 0;
        return attrs.TryGetValue(key, out var raw)
            && double.TryParse(Convert.ToString(raw, CultureInfo.InvariantCulture),
                NumberStyles.Float, CultureInfo.InvariantCulture, out value) && double.IsFinite(value);
    }
}
