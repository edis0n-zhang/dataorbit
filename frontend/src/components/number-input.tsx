import type React from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface NumberInputProps {
  label: string;
  name: string;
  value: number | null;
  onChange: (name: string, value: number | null) => void;
}

export function NumberInput({
  label,
  name,
  value,
  onChange,
}: NumberInputProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue =
      e.target.value === "" ? null : Number.parseInt(e.target.value, 10);
    onChange(name, newValue);
  };

  return (
    <div>
      <Label htmlFor={name}>{label}</Label>
      <Input
        type="number"
        id={name}
        name={name}
        value={value === null ? "" : value}
        onChange={handleChange}
      />
    </div>
  );
}
