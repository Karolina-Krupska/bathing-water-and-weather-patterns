import os, tarfile, gzip, numpy as np, csv, struct, array, re, tempfile
import numpy.ma as ma
from pyproj import CRS, Transformer

# === Radar TAR archive path ===
archive_path = r"C:\Users\earth\OneDrive - University of Reading\REGIMES NEAL DATA PROCESSING\metoffice-c-band-rain-radar_uk_20120102_1km-composite.dat.gz (1).tar"

# === Region bounding boxes
REGIONS_ll = {
    "SEE": ((50, 52), (-2, 2)),
    "EE": ((52, 55), (-2, 2)),
    "SWEW": ((49, 52), (-6, -2)),
    "NWEW": ((52, 55), (-6, -2)),
    "WSNI": ((54, 61), (-8, -4)),
    "ES": ((55, 61), (-4, 0)),
    "Devon": ((50.2, 51.3), (-4.9, -2.8)),
    "Cornwall": ((49.9, 50.7), (-5.7, -4.0)),
}

# === Rainfall bin definitions
rain_bins = [0, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
log2_bins = np.arange(-4, 9, 1)

# === CRS transformation
osgb36 = CRS.from_epsg(27700)
wgs84 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(osgb36, wgs84, always_xy=True)

# === Read radar file and return masked rainfall array
def read_radar_file(file_path, mask_zero=True):
    with open(file_path, "rb") as f:
        struct.unpack(">l", f.read(4))
        gen_ints = array.array("h"); gen_ints.fromfile(f, 31); gen_ints.byteswap()
        gen_reals = array.array("f"); gen_reals.fromfile(f, 28); gen_reals.byteswap()
        spec_reals = array.array("f"); spec_reals.fromfile(f, 45); spec_reals.byteswap()
        f.read(56)
        spec_ints = array.array("h"); spec_ints.fromfile(f, 51); spec_ints.byteswap()
        f.read(4)
        array_size = gen_ints[15] * gen_ints[16]
        struct.unpack(">l", f.read(4))
        data = array.array("h"); data.fromfile(f, array_size); data.byteswap()
        struct.unpack(">l", f.read(4))

        x = np.arange(gen_reals[4], spec_reals[5], gen_reals[5]) / 1000.0
        y = np.flipud(np.arange(gen_reals[2], spec_reals[4], -gen_reals[3])) / 1000.0
        R = np.flipud(np.reshape(data, (gen_ints[15], gen_ints[16]))) / 32.0

        R = ma.masked_where(R <= 0, R) if not mask_zero else ma.masked_where(R < 0, R)
        return R, x, y

# === Process TAR archive for linear or log2 histogram
def process_tar(archive_path, log2=False):
    bins = log2_bins if log2 else rain_bins
    results = {region: [0] * len(bins) for region in REGIONS_ll}
    lat_grid, lon_grid = None, None

    with tarfile.open(archive_path, "r") as tar:
        members = sorted([m for m in tar.getmembers() if m.name.endswith(".gz")], key=lambda m: m.name)
        for member in members:
            print(f"Processing {member.name}...")
            with tar.extractfile(member) as gz_file:
                if gz_file is None:
                    continue
                with gzip.GzipFile(fileobj=gz_file) as f_in:
                    temp_filename = os.path.join(tempfile.gettempdir(), f"temp_{os.path.basename(member.name)}.dat")
                    with open(temp_filename, "wb") as f_out:
                        f_out.write(f_in.read())

                    try:
                        R, x, y = read_radar_file(temp_filename, mask_zero=not log2)
                        if lat_grid is None or lon_grid is None:
                            X, Y = np.meshgrid(x * 1000, y * 1000)
                            lon_grid, lat_grid = transformer.transform(X, Y)

                        for region, (lat_b, lon_b) in REGIONS_ll.items():
                            mask = (lat_grid >= lat_b[0]) & (lat_grid <= lat_b[1]) & \
                                   (lon_grid >= lon_b[0]) & (lon_grid <= lon_b[1])
                            region_data = R[mask].compressed()
                            if region_data.size == 0:
                                continue

                            data_vals = np.log2(region_data) if log2 else region_data
                            for i in range(len(bins)):
                                if i == len(bins) - 1:
                                    count = np.sum(data_vals >= bins[i])
                                else:
                                    count = np.sum((data_vals >= bins[i]) & (data_vals < bins[i + 1]))
                                results[region][i] += int(count)
                    except Exception as e:
                        print(f"Error processing {member.name}: {e}")
                    finally:
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
    return results

# === Save to CSV
def save_results(results, output_csv, bin_labels, filter_region=None):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region"] + bin_labels)
        for region, counts in results.items():
            if filter_region is None or region in filter_region:
                writer.writerow([region] + counts)
    print(f"✅ Saved: {output_csv}")

# === MAIN ===
if __name__ == "__main__":
    # Extract date string from filenames
    with tarfile.open(archive_path, "r") as tar:
        members = sorted([m for m in tar.getmembers() if m.name.endswith(".gz")], key=lambda m: m.name)
        match = re.search(r"\d{8}", os.path.basename(members[0].name)) if members else None
        frame_date = match.group(0) if match else "unknown"

    # Process both types
    linear_results = process_tar(archive_path, log2=False)
    log2_results = process_tar(archive_path, log2=True)

    # Save all 6 files
    save_results(
        linear_results,
        f"rainfall_linear_all_regions_{frame_date}.csv",
        bin_labels=[str(b) for b in rain_bins],
    )
    save_results(
        log2_results,
        f"rainfall_log2_all_regions_{frame_date}.csv",
        bin_labels=[f"log2>={b}" for b in log2_bins],
    )
    
