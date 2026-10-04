using TruckLib;

sealed record PartAttributeRange(int part_index, int start, int end);
sealed record PartAttributeEvidence(string status, int pmd_version = 0,
    int attribute_count = 0, PartAttributeRange[] ranges = null);

static class PartAttributes
{
    public static PartAttributeEvidence Read(string path, IFileSystem files,
        int expectedParts, int expectedAttributes)
    {
        try
        {
            using var stream = files.Open(path);
            if (!stream.CanSeek || stream.Length < 64 || stream.Length > 64*1024*1024)
                return new("unsupported_stream");
            using var reader = new BinaryReader(stream);
            if (reader.ReadUInt32() != 4) return new("unsupported_pmd_version");
            stream.Position = 20;
            uint parts = reader.ReadUInt32(), attributes = reader.ReadUInt32();
            stream.Position = 44;
            uint offset = reader.ReadUInt32();
            if (parts != expectedParts || attributes != expectedAttributes || parts > 4096
                || attributes > 65536 || offset < 64 || (long)offset+parts*8 > stream.Length)
                return new("invalid_attribute_table");
            stream.Position = offset;
            var ranges = new PartAttributeRange[parts];
            var assigned = new bool[attributes];
            for (int i = 0; i < parts; i++)
            {
                int start = reader.ReadInt32(), end = reader.ReadInt32();
                if (start < 0 || end < start || end > attributes)
                    return new("invalid_attribute_range");
                for (int index = start; index < end; index++)
                {
                    if (assigned[index]) return new("overlapping_attribute_ranges");
                    assigned[index] = true;
                }
                ranges[i] = new(i, start, end);
            }
            if (assigned.Any(value => !value)) return new("incomplete_attribute_ranges");
            return new("read", 4, (int)attributes, ranges);
        }
        catch (Exception) { return new("unreadable_attribute_table"); }
    }
}
