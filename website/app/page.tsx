import Hero from "../components/Hero";
import Abstract from "../components/Abstract";
import RelatedWorks from "../components/RelatedWorks";
import Methodology from "../components/Methodology";
import Experiments from "../components/Experiments";
import Results from "../components/Results";
import Conclusion from "../components/Conclusion";
import Team from "../components/Team";
import Links from "../components/Links";

export default function Home() {
  return (
    <>
      <Hero />
      <Abstract />
      <RelatedWorks />
      <Methodology />
      <Experiments />
      <Results />
      <Conclusion />
      <Team />
      <Links />
    </>
  );
}
