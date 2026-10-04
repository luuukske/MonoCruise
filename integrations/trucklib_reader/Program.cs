using System.Numerics;
using System.Text;
using System.Text.Json;
using TruckLib;
using TruckLib.HashFs;
using TruckLib.ScsMap;

if (args.Length != 3) throw new ArgumentException("Expected game folder, target JSON, game version");
using var files = new MountedArchives(args[0]);
var map = new ProgressMap();
map.Read("/map/europe", files);
var widths = new RoadWidths(files);
var templates = new TemplateProfiles(files);
var roads = new List<object>();
int boundaryCount = 0;
foreach (var road in map.MapItems.Values.OfType<Road>())
{
    if (road.Node == null || road.ForwardNode == null) continue;
    var a = road.Node.Position;
    var b = road.ForwardNode.Position;
    if (!Valid(a) || !Valid(b)) continue;
    var chord = Vector2.Distance(new(a.X, a.Z), new(b.X, b.Z));
    if (chord < 0.1 || chord > 5000) continue;
    double ar = Angle(road.Node.Rotation), br = Angle(road.ForwardNode.Rotation);
    var count = Math.Max(4, (int)Math.Ceiling(chord / 5));
    var points = new List<double[]>();
    for (int i = 0; i <= count; i++)
    {
        double t = (double)i / count, u = 1-t;
        double x = u*u*u*a.X + 3*u*u*t*(a.X+Math.Cos(ar)*chord/3)
            + 3*u*t*t*(b.X-Math.Cos(br)*chord/3) + t*t*t*b.X;
        double z = u*u*u*a.Z + 3*u*u*t*(a.Z+Math.Sin(ar)*chord/3)
            + 3*u*t*t*(b.Z-Math.Sin(br)*chord/3) + t*t*t*b.Z;
        points.Add([x, z, u*a.Y+t*b.Y]);
    }
    var boundary = widths.Get(road.RoadType.ToString());
    templates.Collect(road.RoadType.ToString());
    if (boundary != null) boundaryCount++;
    roads.Add(new { id = "road:" + road.Uid, kind = "road", lanes = 0,
        road_type = road.RoadType.ToString(), boundary,
        instance_placement = new { source = "trucklib-road-instance",
            variant_override_count = road.Right.VariantOverrides.Count,
            additional_part_count = road.Right.AdditionalParts.Count,
            right_height_offset_m = road.Right.HeightOffset },
        variant_left = road.Left.Variant.ToString(), variant_right = road.Right.Variant.ToString(), points });
}
if (roads.Count == 0) throw new InvalidDataException("No valid ordinary roads; previous map kept");
var target = Path.GetFullPath(args[1]);
Directory.CreateDirectory(Path.GetDirectoryName(target));
var temporary = target + "." + Guid.NewGuid().ToString("N") + ".tmp";
try
{
    using (var stream = File.Create(temporary))
        JsonSerializer.Serialize(stream, new { schema = 1, game = "ETS2", version = args[2],
            source = "trucklib-official", normal_road_count = roads.Count,
            boundary_road_count = boundaryCount, boundary_status = widths.Status,
            template_profiles = templates.Profiles,
            template_models_read = templates.ReadModels,
            template_models_missing = templates.MissingModels,
            template_models_unsupported = templates.UnsupportedModels,
            template_definition_failures = templates.FailedDefinitions, surface_evidence_version = 1, roads });
    File.Move(temporary, target, true);
}
finally { if (File.Exists(temporary)) File.Delete(temporary); }
Console.WriteLine($"Exported {roads.Count} ordinary roads with official TruckLib");
Console.WriteLine($"Road size definitions available for {boundaryCount} roads; others remain unknown");
Console.WriteLine($"Template measurement profiles: {templates.Profiles.Count}; measurements are not drivable boundaries");
Console.WriteLine($"Models read: {templates.ReadModels}; missing: {templates.MissingModels}; unsupported: {templates.UnsupportedModels}");

static bool Valid(Vector3 p) => float.IsFinite(p.X) && float.IsFinite(p.Y) && float.IsFinite(p.Z)
    && Math.Abs(p.X) < 200000 && Math.Abs(p.Z) < 200000 && Math.Abs(p.Y) < 5000;
static double Angle(Quaternion q) => Math.Atan2(-q.Y, q.W)*2-Math.PI/2;

sealed class ProgressMap : Map
{
    protected override void OnSectorLoading(Sector sector, int index, int total)
    {
        if (index % 50 == 0) Console.WriteLine($"Reading sector {index+1}/{total}");
    }
}

sealed class MountedArchives : IFileSystem, IDisposable
{
    private readonly List<IHashFsReader> readers = [];
    public MountedArchives(string game)
    {
        var paths = new[] { "base.scs", "base_map.scs", "base_share.scs", "def.scs" }.Select(n => Path.Combine(game,n))
            .Concat(Directory.GetFiles(game,"dlc_*.scs").OrderBy(p => p, StringComparer.Ordinal));
        foreach (var path in paths.Where(File.Exists))
        {
            Console.WriteLine("Checking archive " + Path.GetFileName(path));
            var reader = HashFsReader.Open(path);
            readers.Add(reader);
        }
        if (readers.Count == 0) throw new InvalidDataException("No ETS2 map archives found");
    }
    public char DirectorySeparator => '/';
    private IFileSystem Find(string path) => readers.Cast<IFileSystem>().Reverse()
        .FirstOrDefault(r => r.FileExists(path)) ?? throw new FileNotFoundException("Map entry missing",path);
    public bool FileExists(string path) => readers.Any(r => ((IFileSystem)r).FileExists(path));
    public bool DirectoryExists(string path) => readers.Any(r => ((IFileSystem)r).DirectoryExists(path));
    public IList<string> GetFiles(string path) => readers.Cast<IFileSystem>()
        .Where(r => r.DirectoryExists(path)).SelectMany(r => r.GetFiles(path)).Distinct().ToList();
    public byte[] ReadAllBytes(string path) => Find(path).ReadAllBytes(path);
    public string ReadAllText(string path) => Find(path).ReadAllText(path);
    public string ReadAllText(string path, Encoding encoding) => Find(path).ReadAllText(path,encoding);
    public Stream Open(string path) => Find(path).Open(path);
    public string GetParent(string path) => path.Contains('/') ? path[..path.LastIndexOf('/')] : null;
    public void Dispose() { foreach (var reader in readers) reader.Dispose(); }
}
