"""Preprocessing toolkit for real-world data (docs/data_landscape.md §3).

Tested tools the Data Adapter calls instead of re-writing pandas logic every time (a small local model writes one
tool call far more reliably than fifty lines of ad-hoc code). Every tool accepts log=PrepLog() and records what it
changed; the record is shown to the user before they approve the data.

  text:    normalize_text, to_number, roc_to_date, is_missing
  table:   find_header_row, apply_header, drop_total_rows, keep_category, find_records, dedupe, wide_to_long,
           admin_level, attach_parent, keep_level, latest_period
  geo:     detect_crs, to_wgs84, fix_swapped, haversine_matrix, distance_to, within_radius, road_matrix, geocode,
           in_area, points_in_polygon, area_polygon, merge_same_place, drop_stray_points
  join:    admin_key, match, nearest, reduce_points
  quality: PrepLog, report
"""
from .geo import (area_polygon, detect_crs, distance_to, fix_swapped, geocode, haversine_matrix, in_area,
                  drop_stray_points, merge_same_place, points_in_polygon, road_matrix, to_wgs84, within_radius)
from .join import admin_key, match, nearest, reduce_points
from .quality import PrepLog, report
from .table import (admin_level, apply_header, attach_parent, dedupe, drop_total_rows, find_header_row, find_records,
                    keep_category, keep_level, latest_period, wide_to_long)
from .text import is_missing, normalize_text, roc_to_date, to_number
from .table import sample_rows

TOOLS_DOC = __doc__
