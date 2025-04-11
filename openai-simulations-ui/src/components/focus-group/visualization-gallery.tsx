"use client";

import Image from "next/image";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";

interface VisualizationGalleryProps {
  images: string[];
}

export function VisualizationGallery({ images }: VisualizationGalleryProps) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {images.map((image, index) => (
        <Dialog key={index}>
          <DialogTrigger asChild>
            <Card className="overflow-hidden cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-2">
                <div className="relative aspect-video">
                  <Image
                    src={image}
                    alt={`Visualization ${index + 1}`}
                    fill
                    className="object-contain"
                  />
                </div>
                <p className="text-center text-sm text-gray-500 mt-2">
                  {image.split('/').pop()?.replace(/\.[^/.]+$/, "").replace(/_/g, " ")}
                </p>
              </CardContent>
            </Card>
          </DialogTrigger>
          <DialogContent className="max-w-4xl">
            <div className="flex justify-center items-center">
              <div className="relative w-full aspect-video">
                <Image
                  src={image}
                  alt={`Visualization ${index + 1}`}
                  fill
                  className="object-contain"
                />
              </div>
            </div>
            <p className="text-center">
              {image.split('/').pop()?.replace(/\.[^/.]+$/, "").replace(/_/g, " ")}
            </p>
          </DialogContent>
        </Dialog>
      ))}
    </div>
  );
}
