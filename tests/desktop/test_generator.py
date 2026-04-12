"""Tests for bantz.desktop.generator — Config file generation (#365)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bantz.desktop.generator import ConfigGenerator, COLORS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gen() -> ConfigGenerator:
    """Default config generator."""
    return ConfigGenerator()


@pytest.fixture
def custom_gen() -> ConfigGenerator:
    """Custom config generator with non-default params."""
    return ConfigGenerator(
        terminal="alacritty",
        terminal_class="custom-bantz",
        bantz_command="/usr/bin/bantz",
        left_ratio=0.55,
        wallpaper="/home/user/wallpaper.png",
        colors={**COLORS, "accent": "#ff5555"},
        monitor_config="DP-1,1920x1080@144,0x0,1",
    )


# ── Hyprland config ──────────────────────────────────────────────────────────

class TestHyprlandConf:
    def test_contains_monitor(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "monitor = ,preferred,auto,1" in conf

    def test_custom_monitor(self, custom_gen: ConfigGenerator) -> None:
        conf = custom_gen.generate_hyprland_conf()
        assert "monitor = DP-1,1920x1080@144,0x0,1" in conf

    def test_contains_window_rules(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "bantz-chat" in conf
        assert "windowrulev2" in conf
        assert "size 60% 100%" in conf

    def test_custom_terminal_class(self, custom_gen: ConfigGenerator) -> None:
        conf = custom_gen.generate_hyprland_conf()
        assert "custom-bantz" in conf

    def test_exec_once_bantz(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "exec-once = kitty --class bantz-chat -e bantz" in conf

    def test_custom_terminal_exec(self, custom_gen: ConfigGenerator) -> None:
        conf = custom_gen.generate_hyprland_conf()
        assert "exec-once = alacritty --class custom-bantz -e /usr/bin/bantz" in conf

    def test_wallpaper_line_when_set(self, custom_gen: ConfigGenerator) -> None:
        conf = custom_gen.generate_hyprland_conf()
        assert "swww img /home/user/wallpaper.png" in conf

    def test_no_wallpaper_when_empty(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "No wallpaper configured" in conf

    def test_keybindings(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "bind = $mod, Return" in conf
        assert "bind = $mod, Q, killactive" in conf
        assert "bind = $mod, B" in conf

    def test_eww_window_rules(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "class:^(eww-.*)$" in conf
        assert "nofocus" in conf

    def test_dwindle_split_ratio(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        # 0.6 / 0.4 = 1.50
        assert "split_width_multiplier = 1.50" in conf

    def test_custom_split_ratio(self, custom_gen: ConfigGenerator) -> None:
        conf = custom_gen.generate_hyprland_conf()
        # 0.55 / 0.45 ≈ 1.22
        assert "split_width_multiplier = 1.22" in conf

    def test_animations_section(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "bezier = bantzSmooth" in conf
        assert "animation = windows" in conf

    def test_decoration_section(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "rounding = 8" in conf
        assert "blur {" in conf

    def test_screenshot_keybind(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "grim" in conf
        assert "slurp" in conf

    def test_waybar_exec_once(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "exec-once = waybar" in conf

    def test_mako_exec_once(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "exec-once = mako" in conf

    def test_eww_exec_once(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_hyprland_conf()
        assert "exec-once = eww daemon" in conf
        assert "bantz-news bantz-calendar bantz-stats" in conf


# ── Kitty config ──────────────────────────────────────────────────────────────

class TestKittyConf:
    def test_font_family(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_kitty_conf()
        assert "JetBrainsMono Nerd Font" in conf

    def test_background_opacity(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_kitty_conf()
        assert "background_opacity" in conf
        assert "0.92" in conf

    def test_colors(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_kitty_conf()
        assert COLORS["fg"] in conf
        assert COLORS["bg"] in conf
        assert COLORS["accent"] in conf

    def test_scrollback(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_kitty_conf()
        assert "scrollback_lines 10000" in conf

    def test_no_decorations(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_kitty_conf()
        assert "hide_window_decorations yes" in conf

    def test_tab_bar(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_kitty_conf()
        assert "tab_bar_style powerline" in conf


# ── Waybar config ─────────────────────────────────────────────────────────────

class TestWaybarConfig:
    def test_valid_json(self, gen: ConfigGenerator) -> None:
        raw = gen.generate_waybar_config()
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_position_top(self, gen: ConfigGenerator) -> None:
        data = json.loads(gen.generate_waybar_config())
        assert data["position"] == "top"

    def test_bantz_status_module(self, gen: ConfigGenerator) -> None:
        data = json.loads(gen.generate_waybar_config())
        assert "custom/bantz-status" in data["modules-right"]
        assert data["custom/bantz-status"]["exec"] == "bantz-widget-status"

    def test_clock_module(self, gen: ConfigGenerator) -> None:
        data = json.loads(gen.generate_waybar_config())
        assert "clock" in data["modules-center"]

    def test_workspaces_module(self, gen: ConfigGenerator) -> None:
        data = json.loads(gen.generate_waybar_config())
        assert "hyprland/workspaces" in data["modules-left"]

    def test_system_modules(self, gen: ConfigGenerator) -> None:
        data = json.loads(gen.generate_waybar_config())
        assert "cpu" in data["modules-right"]
        assert "memory" in data["modules-right"]
        assert "temperature" in data["modules-right"]


# ── Waybar style ──────────────────────────────────────────────────────────────

class TestWaybarStyle:
    def test_is_css(self, gen: ConfigGenerator) -> None:
        css = gen.generate_waybar_style()
        assert "window#waybar" in css
        assert "#workspaces" in css

    def test_bantz_status_styles(self, gen: ConfigGenerator) -> None:
        css = gen.generate_waybar_style()
        assert "#custom-bantz-status" in css
        assert "@keyframes pulse" in css

    def test_uses_theme_colors(self, gen: ConfigGenerator) -> None:
        css = gen.generate_waybar_style()
        assert COLORS["accent"] in css
        assert COLORS["fg"] in css

    def test_tooltip_style(self, gen: ConfigGenerator) -> None:
        css = gen.generate_waybar_style()
        assert "tooltip" in css


# ── Eww yuck ──────────────────────────────────────────────────────────────────

class TestEwwYuck:
    def test_defpoll_variables(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        for var in ("news-data", "weather-data", "calendar-data", "todo-data",
                     "cpu-usage", "ram-usage", "disk-usage", "gpu-data", "net-data"):
            assert f"defpoll {var}" in yuck

    def test_bantz_widget_data_command(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert "bantz-widget-data" in yuck

    def test_three_panels(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert "news-panel" in yuck
        assert "calendar-panel" in yuck
        assert "stats-panel" in yuck

    def test_three_windows(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert "defwindow bantz-news" in yuck
        assert "defwindow bantz-calendar" in yuck
        assert "defwindow bantz-stats" in yuck

    def test_right_side_geometry(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert 'anchor "top right"' in yuck

    def test_not_focusable(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert ":focusable false" in yuck

    def test_calendar_widget(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert "(calendar" in yuck

    def test_progress_bars(self, gen: ConfigGenerator) -> None:
        yuck = gen.generate_eww_yuck()
        assert "(progress" in yuck


# ── Eww scss ──────────────────────────────────────────────────────────────────

class TestEwwScss:
    def test_scss_variables(self, gen: ConfigGenerator) -> None:
        scss = gen.generate_eww_scss()
        assert "$bg:" in scss
        assert "$accent:" in scss
        assert "$fg:" in scss

    def test_panel_class(self, gen: ConfigGenerator) -> None:
        scss = gen.generate_eww_scss()
        assert ".panel {" in scss

    def test_stat_row(self, gen: ConfigGenerator) -> None:
        scss = gen.generate_eww_scss()
        assert ".stat-row" in scss

    def test_progress_styling(self, gen: ConfigGenerator) -> None:
        scss = gen.generate_eww_scss()
        assert "progress {" in scss

    def test_bantz_status_row(self, gen: ConfigGenerator) -> None:
        scss = gen.generate_eww_scss()
        assert ".bantz-status-row" in scss


# ── Mako config ───────────────────────────────────────────────────────────────

class TestMakoConfig:
    def test_basic_settings(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_mako_config()
        assert "layer=overlay" in conf
        assert "anchor=top-right" in conf

    def test_bantz_app_section(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_mako_config()
        assert "[app-name=bantz]" in conf

    def test_urgency_sections(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_mako_config()
        assert "[urgency=low]" in conf
        assert "[urgency=high]" in conf

    def test_font(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_mako_config()
        assert "JetBrainsMono" in conf

    def test_timeout(self, gen: ConfigGenerator) -> None:
        conf = gen.generate_mako_config()
        assert "default-timeout=8000" in conf


# ── generate_all ──────────────────────────────────────────────────────────────

class TestGenerateAll:
    def test_returns_all_files(self, gen: ConfigGenerator) -> None:
        all_files = gen.generate_all()
        expected_keys = {
            "hypr/hyprland.conf",
            "kitty/kitty.conf",
            "waybar/config",
            "waybar/style.css",
            "eww/eww.yuck",
            "eww/eww.scss",
            "mako/config",
        }
        assert set(all_files.keys()) == expected_keys

    def test_all_values_are_nonempty_strings(self, gen: ConfigGenerator) -> None:
        for key, val in gen.generate_all().items():
            assert isinstance(val, str), f"{key} is not a string"
            assert len(val) > 50, f"{key} is too short"


# ── Deploy ────────────────────────────────────────────────────────────────────

class TestDeploy:
    def test_dry_run(self, gen: ConfigGenerator) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = gen.deploy(td, dry_run=True)
            assert len(written) == 7
            # Nothing should be written
            assert not any(p.exists() for p in written)

    def test_actual_deploy(self, gen: ConfigGenerator) -> None:
        with tempfile.TemporaryDirectory() as td:
            written = gen.deploy(td)
            assert len(written) == 7
            for p in written:
                assert p.exists()
                assert p.stat().st_size > 0

    def test_backup_existing(self, gen: ConfigGenerator) -> None:
        with tempfile.TemporaryDirectory() as td:
            # First deploy
            gen.deploy(td)
            # Second deploy should backup
            gen.deploy(td, backup=True)
            # Check for .bak files
            bak_files = list(Path(td).rglob("*.bak"))
            assert len(bak_files) > 0

    def test_no_backup_when_disabled(self, gen: ConfigGenerator) -> None:
        with tempfile.TemporaryDirectory() as td:
            gen.deploy(td)
            gen.deploy(td, backup=False)
            # No .bak files since backup=False just overwrites
            bak_files = list(Path(td).rglob("*.bak"))
            assert len(bak_files) == 0

    def test_creates_subdirectories(self, gen: ConfigGenerator) -> None:
        with tempfile.TemporaryDirectory() as td:
            gen.deploy(td)
            assert (Path(td) / "hypr").is_dir()
            assert (Path(td) / "kitty").is_dir()
            assert (Path(td) / "waybar").is_dir()
            assert (Path(td) / "eww").is_dir()
            assert (Path(td) / "mako").is_dir()
