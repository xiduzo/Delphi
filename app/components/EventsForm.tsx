"use client";

import * as React from "react";
import { useCodes } from "@/hooks/useCodes";
import { usePredict } from "@/hooks/usePredict";
import { EventCombobox } from "@/components/EventCombobox";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { PatientEvent, PredictResponse } from "@/lib/api";
import { Trash2, Plus } from "lucide-react";
import { z } from "zod";
import { useForm, useFieldArray } from "react-hook-form";
import {
  Form,
  FormField,
  FormItem,
  FormControl,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { zodResolver } from "@hookform/resolvers/zod";

const EventSchema = z.object({
  code: z.string().min(1, "Event is required"),
  age: z.coerce
    .number()
    .refine((v) => Number.isFinite(v), { message: "Age must be a number" })
    .min(0, "Min age is 0")
    .max(120, "Max age is 120"),
});

const FormSchema = z.object({
  events: z.array(EventSchema).min(1, "Add at least one event"),
});

type FormValues = z.infer<typeof FormSchema>;

export default function EventsForm() {
  const { filteredCodes, isLoading } = useCodes();
  const predict = usePredict();

  const form = useForm({
    resolver: zodResolver(FormSchema),
    defaultValues: { events: [{ code: "", age: 0 }] },
    mode: "onBlur",
  });

  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: "events",
  });

  const [result, setResult] = React.useState<PredictResponse | null>(null);

  const onSubmit = async (values: FormValues) => {
    const payload: PatientEvent[] = values.events.map((e) => ({
      code: e.code,
      age_at_event: e.age,
    }));
    try {
      const data = await predict.mutateAsync({ patient: payload });
      setResult(data);
      if (data.warnings?.length) {
        data.warnings.forEach((w) => toast.warning(w));
      } else {
        toast.success("Prediction completed");
      }
    } catch (e: any) {
      toast.error(e?.message || "Prediction failed");
    }
  };

  const clearAll = () => {
    form.reset({ events: [{ code: "", age: 0 }] });
    setResult(null);
  };

  return (
    <div className="mx-auto max-w-4xl p-6">
      <Card>
        <CardHeader>
          <CardTitle>Add life events</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Form {...form}>
            <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
              {fields.map((field, index) => (
                <div
                  key={field.id}
                  className="grid grid-cols-12 gap-3 items-center"
                >
                  <div className="col-span-8">
                    <FormField
                      control={form.control}
                      name={`events.${index}.code`}
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Event</FormLabel>
                          <FormControl>
                            <EventCombobox
                              value={field.value || null}
                              onChange={(val) => field.onChange(val ?? "")}
                              options={filteredCodes}
                              loading={isLoading}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                  <div className="col-span-3">
                    <FormField
                      control={form.control}
                      name={`events.${index}.age`}
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Age</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="decimal"
                              min={0}
                              max={120}
                              value={
                                Number.isFinite(field.value as number)
                                  ? String(field.value)
                                  : ""
                              }
                              onChange={(e) =>
                                field.onChange(
                                  e.target.value === ""
                                    ? undefined
                                    : Number(e.target.value)
                                )
                              }
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                  <div className="col-span-1 flex justify-end">
                    <Button
                      variant="ghost"
                      size="icon"
                      type="button"
                      onClick={() => remove(index)}
                      aria-label="Remove"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}

              <div>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => append({ code: "", age: 0 })}
                >
                  <Plus className="mr-2 h-4 w-4" /> Add event
                </Button>
              </div>

              <div className="flex gap-2">
                <Button type="submit" disabled={predict.isPending}>
                  Submit
                </Button>
                <Button type="button" variant="outline" onClick={clearAll}>
                  Clear
                </Button>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>

      {result && (
        <div className="mt-8">
          <h2 className="text-lg font-semibold mb-2">Top predictions</h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[120px]">Code</TableHead>
                <TableHead>Label</TableHead>
                <TableHead className="w-[160px] text-right">
                  Probability
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {result.ranked.map((r) => (
                <TableRow key={r.index}>
                  <TableCell className="font-mono">{r.code ?? "-"}</TableCell>
                  <TableCell>{r.label ?? "-"}</TableCell>
                  <TableCell className="text-right">
                    {(r.probability * 100).toFixed(2)}%
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {result.warnings?.length ? (
            <div className="mt-4 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
              {result.warnings.map((w, i) => (
                <div key={i} className="py-0.5">
                  {w}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
