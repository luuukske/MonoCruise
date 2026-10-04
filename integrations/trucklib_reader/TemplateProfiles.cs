using System.Numerics;
using System.Globalization;
using TruckLib;
using TruckLib.Sii;
using MeshModel = TruckLib.Models.Model;

sealed class TemplateProfiles
{
    private readonly IFileSystem files;
    private readonly Dictionary<string, Unit> definitions = new();
    private readonly Dictionary<string, object> models = new();
    public Dictionary<string, object> Profiles { get; } = new();
    public int FailedDefinitions { get; private set; }
    public int ReadModels { get; private set; }
    public int MissingModels { get; private set; }
    public int UnsupportedModels { get; private set; }

    public TemplateProfiles(IFileSystem files)
    {
        this.files = files;
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
                    definitions[key] = unit;
                }
            }
            catch (Exception) { FailedDefinitions++; }
        }
    }

    public void Collect(string token)
    {
        if (Profiles.ContainsKey(token) || !definitions.TryGetValue(token, out var unit)) return;
        var sides = new Dictionary<string, object>();
        foreach (var side in new[] { "left", "right" })
        {
            if (!unit.Attributes.TryGetValue("template_"+side, out var raw) || raw is not string path) continue;
            if (string.IsNullOrWhiteSpace(path)) continue;
            if (!models.TryGetValue(path, out var profile))
            {
                profile = Read(path);
                models[path] = profile;
            }
            sides[side] = profile;
        }
        if (sides.Count > 0) Profiles[token] = new { source = "trucklib-model-measurements",
            drivable_boundary = false, placement = Placement(unit), templates = sides };
    }

    private static object Placement(Unit unit)
    {
        var attrs = unit.Attributes;
        bool present = attrs.TryGetValue("road_offset", out var raw);
        string text = Convert.ToString((object)raw, CultureInfo.InvariantCulture);
        bool parsed = double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number);
        double? offset = present && parsed && double.IsFinite(number)
            ? number : null;
        var extra = attrs.Keys.Where(key => key.Contains("offset", StringComparison.Ordinal)
            && key != "road_offset").Order().ToArray();
        return new { source = "sii-road-look", road_offset_present = present,
            road_offset_m = offset, unresolved_offset_keys = extra };
    }

    private object Read(string path)
    {
        try
        {
            if (!files.FileExists(path) || !files.FileExists(Path.ChangeExtension(path, ".pmg")))
            {
                MissingModels++;
                return new { status = "model_files_missing" };
            }
            Console.WriteLine($"Reading road template model {models.Count+1}");
            var model = MeshModel.Open(path, files);
            var parts = model.Parts.Select(part => new {
                name = part.Name.ToString(),
                pieces = part.Pieces.Select(piece => new {
                    material_slot = piece.Material,
                    geometry = Measure(piece.Vertices.Select(v => v.Position).ToArray())
                }).ToArray(),
                locators = part.Locators.Take(256).Where(l => Finite(l.Position)).Select(l => new {
                    name = l.Name.ToString(), position = new[] { l.Position.X, l.Position.Y, l.Position.Z }
                }).ToArray()
            }).ToArray();
            var variants = model.Variants.Select(v => new { name = v.Name.ToString(),
                attributes = v.Attributes.Select(a => new { tag = a.Tag.ToString(), a.Type, a.Value }).ToArray()
            }).ToArray();
            var materials = model.Looks.Select(l => new { name = l.Name.ToString(),
                slots = l.Materials.ToArray() }).ToArray();
            var part_attributes = PartAttributes.Read(path, files, model.Parts.Count,
                model.Variants.Count > 0 ? model.Variants[0].Attributes.Count : 0);
            ReadModels++;
            return new { status = "read", parts, variants, materials, part_attributes };
        }
        catch (UnsupportedVersionException)
        {
            UnsupportedModels++;
            return new { status = "unsupported_model_version" };
        }
        catch (Exception) { return new { status = "unreadable_model" }; }
    }

    private static object Measure(Vector3[] input)
    {
        var vertices = input.Where(Finite).ToArray();
        if (vertices.Length == 0) return new { status = "no_finite_vertices" };
        double low = vertices.Min(v => v.Z), high = vertices.Max(v => v.Z);
        double step = Math.Max(0.001, (high-low)/16);
        var sections = vertices.GroupBy(v => Math.Min(15, (int)((v.Z-low)/step)))
            .OrderBy(g => g.Key).Select(g => new {
                min_z = g.Min(v => v.Z), max_z = g.Max(v => v.Z),
                min_x = g.Min(v => v.X), max_x = g.Max(v => v.X),
                min_y = g.Min(v => v.Y), max_y = g.Max(v => v.Y)
            }).ToArray();
        return new { status = "measured", vertex_count = vertices.Length,
            min_x = vertices.Min(v => v.X), max_x = vertices.Max(v => v.X),
            min_y = vertices.Min(v => v.Y), max_y = vertices.Max(v => v.Y),
            min_z = low, max_z = high, sections };
    }

    private static bool Finite(Vector3 v) => float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z)
        && Math.Abs(v.X) < 10000 && Math.Abs(v.Y) < 10000 && Math.Abs(v.Z) < 10000;
}
