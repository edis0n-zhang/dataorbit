"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/number-input";
import { TagInput } from "@/components/tag-input";

export default function HealthForm() {
  const [formData, setFormData] = useState({
    race: null,
    age: null,
    ethnicity: null,
    gender: null,
    height: null,
    weight: null,
    bmi: null,
    diastolicBP: null,
    systolicBP: null,
    heartRate: null,
    respiratoryRate: null,
    conditions: [],
  });

  const handleInputChange = (name: string, value: number | null) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSelectChange = (name: string) => (value: string) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleConditionsChange = (conditions: string[]) => {
    setFormData((prev) => ({ ...prev, conditions }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Raw form data:", formData);
    const encodedData = {
      race: formData.race
        ? ["white", "black", "asian", "hawaiian", "native", "other"].indexOf(
            formData.race
          )
        : null,
      age: formData.age,
      ethnicity:
        formData.ethnicity === "hispanic"
          ? 1
          : formData.ethnicity === "nonhispanic"
            ? 0
            : null,
      gender: formData.gender === "M" ? 1 : formData.gender === "F" ? 0 : null,
      height: formData.height,
      weight: formData.weight,
      bmi: formData.bmi,
      diastolicBP: formData.diastolicBP,
      systolicBP: formData.systolicBP,
      heartRate: formData.heartRate,
      respiratoryRate: formData.respiratoryRate,
      conditions: formData.conditions,
    };

    console.log("Encoded data for submission:", encodedData);

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/api/submit-health-data",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(encodedData),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.message || "Failed to submit data");
      }

      const result = await response.json();
      alert("Data submitted successfully");
      console.log("Prediction result:", result);
    } catch (error) {
      console.error("Error:", error);
      alert(error instanceof Error ? error.message : "Error submitting data");
    }
  };

  return (
    <div className="container mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Health Information Form</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Label htmlFor="race">Race</Label>
              <Select onValueChange={handleSelectChange("race")}>
                <SelectTrigger id="race">
                  <SelectValue placeholder="Select race" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="white">White</SelectItem>
                  <SelectItem value="black">Black</SelectItem>
                  <SelectItem value="asian">Asian</SelectItem>
                  <SelectItem value="hawaiian">Hawaiian</SelectItem>
                  <SelectItem value="native">Native</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <NumberInput
              label="Age"
              name="age"
              value={formData.age}
              onChange={handleInputChange}
            />

            <div>
              <Label htmlFor="ethnicity">Ethnicity</Label>
              <Select onValueChange={handleSelectChange("ethnicity")}>
                <SelectTrigger id="ethnicity">
                  <SelectValue placeholder="Select ethnicity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hispanic">Hispanic</SelectItem>
                  <SelectItem value="nonhispanic">Non-Hispanic</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label>Gender</Label>
              <RadioGroup onValueChange={handleSelectChange("gender")}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="M" id="male" />
                  <Label htmlFor="male">Male</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="F" id="female" />
                  <Label htmlFor="female">Female</Label>
                </div>
              </RadioGroup>
            </div>

            <NumberInput
              label="Body Height (cm)"
              name="height"
              value={formData.height}
              onChange={handleInputChange}
            />

            <NumberInput
              label="Body Weight (kg)"
              name="weight"
              value={formData.weight}
              onChange={handleInputChange}
            />

            <NumberInput
              label="Body Mass Index (kg/m²)"
              name="bmi"
              value={formData.bmi}
              onChange={handleInputChange}
            />

            <NumberInput
              label="Diastolic Blood Pressure (mm[Hg])"
              name="diastolicBP"
              value={formData.diastolicBP}
              onChange={handleInputChange}
            />

            <NumberInput
              label="Systolic Blood Pressure (mm[Hg])"
              name="systolicBP"
              value={formData.systolicBP}
              onChange={handleInputChange}
            />

            <NumberInput
              label="Heart Rate (/min)"
              name="heartRate"
              value={formData.heartRate}
              onChange={handleInputChange}
            />

            <NumberInput
              label="Respiratory Rate (/min)"
              name="respiratoryRate"
              value={formData.respiratoryRate}
              onChange={handleInputChange}
            />

            <TagInput
              label="Conditions"
              tags={formData.conditions}
              onChange={handleConditionsChange}
            />

            <Button type="submit">Submit</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
