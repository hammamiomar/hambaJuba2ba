import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";

export type Mode = "looping" | "audio" | "fourcorner";

interface ModeSelectorProps {
  currentMode: Mode;
  onModeChange: (mode: Mode) => void;
  disabled?: boolean;
}

const MODE_LABELS: Record<Mode, string> = {
  looping: "Looping",
  audio: "Audio Reactive",
  fourcorner: "Four-Corner",
};

export function ModeSelector({
  currentMode,
  onModeChange,
  disabled = false,
}: ModeSelectorProps) {
  return (
    <Menu>
      <MenuButton
        disabled={disabled}
        className="inline-flex w-full items-center justify-between gap-2 rounded-md bg-[#8B9A7E]/20 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-[#8B9A7E]/20 focus:outline-none data-focus:outline data-focus:outline-1 data-focus:outline-[#8B9A7E] data-hover:bg-[#8B9A7E]/30 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <span>Mode: {MODE_LABELS[currentMode]}</span>
        <svg className="size-4 fill-white/60" viewBox="0 0 16 16" fill="none">
          <path d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z" />
        </svg>
      </MenuButton>

      <MenuItems
        transition
        anchor="bottom"
        className="z-50 w-52 origin-top rounded-xl border border-[#8B9A7E]/20 bg-[#8B9A7E]/10 p-1 text-sm/6 text-white shadow-xl backdrop-blur-lg transition duration-100 ease-out [--anchor-gap:4px] focus:outline-none data-closed:scale-95 data-closed:opacity-0"
      >
        <MenuItem>
          <button
            onClick={() => onModeChange("looping")}
            className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-[#8B9A7E]/20"
          >
            {currentMode === "looping" && (
              <svg className="size-4 fill-[#B5CC9A]" viewBox="0 0 16 16">
                <circle cx="8" cy="8" r="3" />
              </svg>
            )}
            <span className={currentMode === "looping" ? "ml-0" : "ml-6"}>
              Looping
            </span>
          </button>
        </MenuItem>

        <MenuItem>
          <button
            onClick={() => onModeChange("audio")}
            className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-[#8B9A7E]/20"
          >
            {currentMode === "audio" && (
              <svg className="size-4 fill-[#B5CC9A]" viewBox="0 0 16 16">
                <circle cx="8" cy="8" r="3" />
              </svg>
            )}
            <span className={currentMode === "audio" ? "ml-0" : "ml-6"}>
              Audio Reactive
            </span>
          </button>
        </MenuItem>

        <MenuItem>
          <button
            onClick={() => onModeChange("fourcorner")}
            className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-[#8B9A7E]/20"
          >
            {currentMode === "fourcorner" && (
              <svg className="size-4 fill-[#B5CC9A]" viewBox="0 0 16 16">
                <circle cx="8" cy="8" r="3" />
              </svg>
            )}
            <span className={currentMode === "fourcorner" ? "ml-0" : "ml-6"}>
              Four-Corner
            </span>
          </button>
        </MenuItem>
      </MenuItems>
    </Menu>
  );
}
