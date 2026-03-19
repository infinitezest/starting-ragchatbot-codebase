# Frontend Changes: Dark/Light Mode Toggle

## Overview
Added a theme toggle button that switches between dark and light modes with smooth transitions, localStorage persistence, and full keyboard accessibility.

## Files Modified

### `frontend/index.html`
- Added inline `<script>` in `<head>` to apply saved theme before first paint (prevents flash of wrong theme)
- Added theme toggle button with inline SVG sun/moon icons, positioned inside `.container` before `.main-content`
- Bumped cache-busting version from `?v=9` to `?v=10` on CSS and JS links

### `frontend/style.css`
- Added `[data-theme="light"]` CSS variable overrides for the light color palette (Tailwind Slate scale: `#f8fafc` background, `#ffffff` surface, `#0f172a` text)
- Added light-theme overrides for hardcoded colors: assistant message links, source chips, code blocks, error/success messages, welcome message shadow
- Added `.theme-toggle` button styles: 40px fixed-position circle in top-right, hover lift effect, focus ring
- Added `.theme-icon` animation: sun/moon cross-fade with rotation (0.3s transitions)
- Added global theme transition rule (`background-color`, `color`, `border-color` at 0.3s) for smooth switching
- Added responsive override at 768px breakpoint: button shrinks to 36px with tighter margins

### `frontend/script.js`
- Added IIFE at top of file to read `localStorage('theme')` and set `data-theme` attribute before DOM is ready
- Added `toggleTheme()` function: toggles `data-theme="light"` attribute, updates localStorage, updates `aria-label`
- Added event listener for the theme toggle button inside `DOMContentLoaded`

## Design Decisions
- **Dark is default**: No `data-theme` attribute = dark theme (existing behavior preserved)
- **Light is opt-in**: `data-theme="light"` overrides CSS variables
- **Primary blue (#2563eb) stays the same** in both themes for brand consistency
- **Flash prevention**: Inline script in `<head>` runs before stylesheet evaluation
- **Icon convention**: Sun icon in dark mode (click to get light), moon icon in light mode (click to get dark)

## Light Theme Refinement
- Changed `.message.assistant .message-content` background from `var(--surface)` to `var(--assistant-message)` so assistant bubbles are visually distinct from the page background in light mode (`#e2e8f0` on `#f8fafc`)
- Changed `.message.welcome-message .message-content` background from `var(--surface)` to `var(--welcome-bg)` and border from `var(--border-color)` to `var(--welcome-border)` for a light blue tinted welcome message in light mode (`#eff6ff`)
- Added `.main-content`, `.source-chip`, and `.course-title-item` to the smooth theme transition rule so all visible elements animate during toggle

## Implementation Details
- **CSS custom properties**: All theme colors are defined as CSS variables in `:root` (dark) and `[data-theme="light"]` (light), enabling a single-attribute switch
- **`data-theme` attribute**: Set on `<html>` element via `document.documentElement.setAttribute()`, detected by CSS `[data-theme="light"]` selectors
- **Toggle via JS**: `toggleTheme()` flips the attribute and persists to `localStorage`; IIFE at page load restores it before first paint
- **Smooth transitions**: `background-color`, `color`, and `border-color` transitions (0.3s ease) applied to all themed elements; icon swap uses opacity + rotate (0.3s)
- **Visual hierarchy preserved**: Primary blue accent unchanged, contrast ratios meet WCAG AA, surface/background distinction maintained in both themes
