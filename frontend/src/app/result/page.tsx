"use client";

import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ResultPage() {
  const searchParams = useSearchParams();
  const prediction = searchParams.get("prediction");
  
  const formattedPrediction = prediction 
    ? new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(Number(prediction))
    : "N/A";

  return (
    <div className="container mx-auto p-4">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl text-center">Result</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-lg mb-4">
            Your predicted out of pocket spending for the rest of your life is:
          </p>
          <p className="text-4xl font-bold text-center text-primary">
            {formattedPrediction}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
